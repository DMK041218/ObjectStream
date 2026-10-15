"""CPU-sized end-to-end generation with real visual compression and KV caching."""
import pytest
import torch

from qwen2_5_vl_objectstream import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration


def tiny_model(attention):
    torch.manual_seed(17)
    config = Qwen2_5_VLConfig(
        vocab_size=128, hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=512,
        rope_scaling={"type": "mrope", "mrope_section": [2, 1, 1]},
        vision_config=dict(depth=1, hidden_size=32, intermediate_size=64,
            num_heads=4, patch_size=2, spatial_merge_size=2,
            temporal_patch_size=2, window_size=8, out_hidden_size=32,
            fullatt_block_indexes=[0]),
        image_token_id=113, video_token_id=112, vision_start_token_id=110,
        vision_end_token_id=111, bos_token_id=1, eos_token_id=2, pad_token_id=0,
    )
    config._attn_implementation = attention
    return Qwen2_5_VLForConditionalGeneration(config).eval()


def video_inputs(groups=12):
    ids = torch.tensor([[1, 110] + [112] * (groups * 4) + [111, 5, 6]])
    generator = torch.Generator().manual_seed(29)
    return dict(input_ids=ids, attention_mask=torch.ones_like(ids),
        pixel_values_videos=torch.randn(groups * 16, 24, generator=generator),
        video_grid_thw=torch.tensor([[groups, 4, 4]]), second_per_grid_ts=[2.0])


def generate(model, inputs, enabled=True, **kwargs):
    with torch.inference_mode():
        return model.generate(**inputs, max_new_tokens=3, min_new_tokens=3,
            do_sample=False, use_objectstream=enabled, short_frames=8,
            medium_frames=64, entity_max_entities=64, entity_tokens_per_entity=8,
            entity_saliency_quantile=.6, entity_match_threshold=.72,
            entity_temporal_burst_selection=True, entity_temporal_burst_max_events=32,
            entity_temporal_burst_radius=1, entity_temporal_burst_tokens_per_frame=2,
            return_dict_in_generate=True, output_scores=True, **kwargs)


@pytest.mark.parametrize("attention", ["eager", "sdpa"])
def test_compressed_generation_separates_cache_slots_from_rope(attention):
    model = tiny_model(attention)
    inputs = video_inputs()
    observed = []

    def inspect_positions(module, args, kwargs):
        observed.append((kwargs["cache_position"].clone(), kwargs["position_ids"].clone()))

    hook = model.model.layers[0].register_forward_pre_hook(inspect_positions, with_kwargs=True)
    try:
        output = generate(model, inputs)
    finally:
        hook.remove()
    original_length = inputs["input_ids"].shape[-1]
    compressed_length = len(observed[0][0])
    assert compressed_length < original_length  # Must exercise actual compression.
    assert output.sequences.shape[-1] == original_length + 3
    assert torch.equal(observed[0][0], torch.arange(compressed_length))
    semantic_last = int(observed[0][1].max())
    for step, (cache, positions) in enumerate(observed[1:], 1):
        assert cache.tolist() == [compressed_length + step - 1]
        assert torch.all(positions == semantic_last + step)


def test_eager_and_sdpa_compressed_generation_agree():
    inputs = video_inputs()
    eager = generate(tiny_model("eager"), inputs)
    sdpa = generate(tiny_model("sdpa"), inputs)
    assert torch.equal(eager.sequences, sdpa.sequences)
    for left, right in zip(eager.scores, sdpa.scores):
        torch.testing.assert_close(left, right, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("attention", ["eager", "sdpa"])
def test_short_video_preserves_native_generation(attention):
    inputs = video_inputs(groups=4)
    native = generate(tiny_model(attention), inputs, enabled=False)
    memory = generate(tiny_model(attention), inputs)
    assert torch.equal(native.sequences, memory.sequences)
    for left, right in zip(native.scores, memory.scores):
        torch.testing.assert_close(left, right)


@pytest.mark.parametrize("attention", ["eager", "sdpa"])
def test_compressed_generation_without_cache(attention):
    inputs = video_inputs()
    output = generate(tiny_model(attention), inputs, use_cache=False)
    assert output.sequences.shape[-1] == inputs["input_ids"].shape[-1] + 3
