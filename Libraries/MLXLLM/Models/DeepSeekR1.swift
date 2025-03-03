//
//  DeepSeek-R1-Distill-Qwen-1.5B-4bit.swift
//  mlx-swift-examples
//
//  Created by everettjf on 2/1/25.
//
import Foundation
import MLX
import MLXFast
import MLXLMCommon
import MLXNN

public struct DeepSeekR1Configuration: Codable, Sendable {
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let numAttentionHeads: Int
    public let numHiddenLayers: Int
    public let numKeyValueHeads: Int
    public let vocabularySize: Int
    public let slidingWindow: Int
    public let maxPositionEmbeddings: Int
    
    // Values with defaults
    public let _rmsNormEps: Float?
    public var rmsNormEps: Float { _rmsNormEps ?? 1e-6 }
    
    public let _attentionDropout: Float?
    public var attentionDropout: Float { _attentionDropout ?? 0.0 }
    
    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numAttentionHeads = "num_attention_heads"
        case numHiddenLayers = "num_hidden_layers"
        case numKeyValueHeads = "num_key_value_heads"
        case vocabularySize = "vocab_size"
        case slidingWindow = "sliding_window"
        case maxPositionEmbeddings = "max_position_embeddings"
        case _rmsNormEps = "rms_norm_eps"
        case _attentionDropout = "attention_dropout"
    }
}

private class Attention: Module {
    let config: DeepSeekR1Configuration
    let scale: Float

    let heads: Int
    let kvHeads: Int
    let headDim: Int

    @ModuleInfo(key: "qkv_proj") var wqkv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear
    
    let rope: RoPE

    public init(_ config: DeepSeekR1Configuration) {
        self.config = config
        
        let dim = config.hiddenSize
        self.heads = config.numAttentionHeads
        self.kvHeads = config.numKeyValueHeads
        
        self.headDim = config.hiddenSize / heads
        self.scale = pow(Float(headDim), -0.5)
        
        self._wqkv.wrappedValue = Linear(dim, (heads + 2 * kvHeads) * headDim, bias: false)
        self._wo.wrappedValue = Linear(heads * headDim, dim, bias: false)
        
        self.rope = RoPE(dimensions: headDim, traditional: true, base: 10000)
    }

    public func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil, cache: KVCache?) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        let queryPos = heads * headDim
        let qkv = split(wqkv(x), indices: [queryPos, queryPos + kvHeads * headDim], axis: -1)
        var queries = qkv[0]
        var keys = qkv[1]
        var values = qkv[2]

        // prepare the queries, keys and values for the attention computation
        queries = queries.reshaped(B, L, heads, -1).transposed(0, 2, 1, 3)
        keys = keys.reshaped(B, L, kvHeads, -1).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, kvHeads, -1).transposed(0, 2, 1, 3)

        if let cache {
            queries = rope.callAsFunction(queries, offset: cache.offset)
            keys = rope.callAsFunction(keys, offset: cache.offset)
            (keys, values) = cache.update(keys: keys, values: values)
        } else {
            queries = rope.callAsFunction(queries)
            keys = rope.callAsFunction(keys)
        }

        let output = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: scale, mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return wo(output)
    }
}

private class MLP: Module, UnaryLayer {

    @ModuleInfo(key: "gate_up_proj") var gate_up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear

    public init(dimensions: Int, hiddenDimensions: Int) {
        self._gate_up.wrappedValue = Linear(dimensions, 2 * hiddenDimensions, bias: false)
        self._down.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let gu = split(gate_up(x), parts: 2, axis: -1)
        return down(silu(gu[0]) * gu[1])
    }
}

private class TransformerBlock: Module {

    @ModuleInfo(key: "self_attn") var attention: Attention
    let mlp: MLP

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    public init(_ config: DeepSeekR1Configuration) {
        self._attention.wrappedValue = Attention(config)
        self.mlp = MLP(dimensions: config.hiddenSize, hiddenDimensions: config.intermediateSize)
        self._inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    public func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil, cache: KVCache?) -> MLXArray {
        var r = attention(inputLayerNorm(x), mask: mask, cache: cache)
        let h = x + r
        r = mlp(postAttentionLayerNorm(h))
        let out = h + r
        return out
    }
}

fileprivate class DeepSeekR1ModelInner: Module {
    
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    fileprivate let layers: [TransformerBlock]
    let norm: RMSNorm

    public init(_ config: DeepSeekR1Configuration) {
        precondition(config.vocabularySize > 0)

        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)

        self.layers = (0 ..< config.numHiddenLayers)
            .map { _ in
                TransformerBlock(config)
            }
        self.norm = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var h = embedTokens(inputs)

        let mask: MLXArray? = createAttentionMask(h: h, cache: cache)

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
        }

        return norm(h)
    }
}

public class DeepSeekR1Model: Module, LLMModel, KVCacheDimensionProvider, LoRAModel {
    public let kvHeads: [Int]
    @ModuleInfo fileprivate var model: DeepSeekR1ModelInner
    
    public init(_ args: DeepSeekR1Configuration) {
        self.kvHeads = Array(repeating: args.numKeyValueHeads, count: args.numHiddenLayers)
        self.model = DeepSeekR1ModelInner(args)
    }
    
    public func loraLinearLayers() -> LoRALinearLayers {
        model.layers.map { ($0.attention, ["q_proj", "v_proj"]) }
    }
    
    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let out = model(inputs, cache: cache)
        return model.embedTokens.asLinear(out)
    }
}
