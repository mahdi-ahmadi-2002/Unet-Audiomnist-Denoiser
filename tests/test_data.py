import numpy as np
import pytest
import torch

from denoiser.data import NoisyDigits, build_datasets, load_speaker_metadata, scan, split_by_speaker


def test_scan_pairs_clean_and_noisy(corpus):
    utterances = scan(corpus)
    assert len(utterances) == 8 * 2 * 2
    assert all(u.noisy_path and u.noisy_path.endswith("_noisy.wav") for u in utterances)
    assert {u.digit for u in utterances} == {0, 1}


def test_scan_missing_folder_explains_itself(tmp_path):
    with pytest.raises(FileNotFoundError, match="README"):
        scan(tmp_path / "nope")


def test_split_has_no_speaker_overlap_and_is_deterministic(corpus):
    utterances = scan(corpus)
    a = split_by_speaker(utterances, 2, 2, seed=0)
    b = split_by_speaker(utterances, 2, 2, seed=0)
    speakers = {name: {u.speaker for u in part} for name, part in a.items()}
    assert [len(speakers[k]) for k in ("train", "val", "test")] == [4, 2, 2]
    assert not speakers["train"] & speakers["val"]
    assert not speakers["train"] & speakers["test"]
    assert not speakers["val"] & speakers["test"]
    assert a == b
    assert sum(len(p) for p in a.values()) == len(utterances)


def test_items_have_expected_shape_and_snr(cfg):
    item = build_datasets(cfg)["test"][0]
    assert item["noisy"].shape == item["clean"].shape == (cfg.audio.segment_samples,)
    assert item["noisy"].dtype == torch.float32
    assert cfg.data.snr_min <= item["snr_db"] <= cfg.data.snr_max
    assert item["noise_type"] in cfg.data.noise_types
    assert float(item["noisy"].abs().max()) <= 1.0


def test_synthetic_snr_is_measured_on_the_speech_not_the_padding(cfg):
    cfg.data.noise_types = ["white"]
    item = build_datasets(cfg)["test"][0]
    clean, noise = item["clean"].numpy(), (item["noisy"] - item["clean"]).numpy()
    active = clean != 0
    measured = 10 * np.log10(np.mean(clean[active] ** 2) / np.mean(noise**2))
    assert measured == pytest.approx(item["snr_db"], abs=0.5)


def test_eval_split_is_reproducible_but_train_is_random(cfg):
    datasets = build_datasets(cfg)
    assert torch.equal(datasets["test"][3]["noisy"], datasets["test"][3]["noisy"])
    assert not torch.equal(datasets["train"][3]["noisy"], datasets["train"][3]["noisy"])


def test_every_noise_type_yields_finite_audio(cfg):
    utterances = scan(cfg.data.root)
    for kind in ("white", "pink", "brown", "hum", "babble"):
        cfg.data.noise_types = [kind]
        item = NoisyDigits(utterances, cfg, train=True)[0]
        assert torch.isfinite(item["noisy"]).all()
        assert not torch.equal(item["noisy"], item["clean"])


def test_paired_mode_uses_the_noisy_twin(cfg):
    cfg.data.noise_source = "paired"
    item = build_datasets(cfg)["test"][0]
    assert item["noise_type"] == "paired"
    assert 0 < item["snr_db"] < 30


def test_metadata_is_found(corpus):
    assert load_speaker_metadata(corpus)["01"]["gender"] == "female"
