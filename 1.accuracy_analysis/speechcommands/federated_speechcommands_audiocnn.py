import argparse
import random
import tarfile
import urllib.request
import wave
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from model.AudioCNN import AudioCNN


SPEECH_COMMANDS_URL = "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
COMMAND_LABELS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simple federated learning on Speech Commands with AudioCNN and Dirichlet partitioning"
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data") / "speech_commands")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--num-clients", type=int, default=50)
    parser.add_argument("--clients-per-round", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--test-batch-size", type=int, default=256)
    parser.add_argument("--dirichlet-alpha", type=float, default=0.5)
    parser.add_argument("--min-samples-per-client", type=int, default=10)
    parser.add_argument("--local-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--clip-duration-ms", type=int, default=1000)
    parser.add_argument("--n-fft", type=int, default=400)
    parser.add_argument("--hop-length", type=int, default=160)
    parser.add_argument("--win-length", type=int, default=400)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(args):
    if torch.cuda.is_available() and 0 <= args.gpu_id < torch.cuda.device_count():
        return torch.device(f"cuda:{args.gpu_id}")
    return torch.device("cpu")


def is_valid_speech_commands_root(path):
    if not path.exists() or not path.is_dir():
        return False
    if not (path / "validation_list.txt").exists():
        return False
    if not (path / "testing_list.txt").exists():
        return False
    return all((path / label_name).exists() for label_name in COMMAND_LABELS)


def find_speech_commands_root(data_dir):
    candidates = [
        data_dir / "SpeechCommands" / "speech_commands_v0.02",
        data_dir / "speech_commands_v0.02",
        data_dir / "SpeechCommands",
        data_dir,
    ]
    for candidate in candidates:
        if is_valid_speech_commands_root(candidate):
            return candidate

    for validation_path in data_dir.rglob("validation_list.txt"):
        candidate = validation_path.parent
        if is_valid_speech_commands_root(candidate):
            return candidate
    return None


def maybe_download_speech_commands(data_dir, should_download):
    existing_root = find_speech_commands_root(data_dir)
    if existing_root is not None:
        return existing_root

    if not should_download:
        print("Speech Commands dataset was not found locally. Trying automatic download now.")

    data_dir.mkdir(parents=True, exist_ok=True)
    archive_path = data_dir / "speech_commands_v0.02.tar.gz"
    try:
        urllib.request.urlretrieve(SPEECH_COMMANDS_URL, archive_path)
        extract_root = data_dir / "SpeechCommands"
        extract_root.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive_path, "r:gz") as tar_handle:
            tar_handle.extractall(extract_root)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download Speech Commands automatically into {data_dir}."
        ) from exc

    resolved_root = find_speech_commands_root(data_dir)
    if resolved_root is None:
        raise FileNotFoundError(
            "Speech Commands download/extraction finished, but the dataset root could not be located under "
            f"{data_dir}."
        )
    return resolved_root


def load_split_file(split_path):
    return {line.strip() for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip()}


def collect_samples(dataset_root, split, max_samples):
    validation_relpaths = load_split_file(dataset_root / "validation_list.txt")
    testing_relpaths = load_split_file(dataset_root / "testing_list.txt")
    samples = []

    for label_idx, label_name in enumerate(COMMAND_LABELS):
        label_dir = dataset_root / label_name
        wav_paths = sorted(label_dir.glob("*.wav"))
        for wav_path in wav_paths:
            relpath = wav_path.relative_to(dataset_root).as_posix()
            if relpath in validation_relpaths:
                sample_split = "validation"
            elif relpath in testing_relpaths:
                sample_split = "testing"
            else:
                sample_split = "training"

            if split == "train" and sample_split != "training":
                continue
            if split == "test" and sample_split == "training":
                continue

            samples.append((wav_path, label_idx))
            if max_samples > 0 and len(samples) >= max_samples:
                return samples
    return samples


def read_waveform(wav_path, target_sample_rate, target_num_samples):
    with wave.open(str(wav_path), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        num_channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        num_frames = wav_file.getnframes()
        audio_bytes = wav_file.readframes(num_frames)

    if sample_width != 2:
        raise ValueError(f"Unsupported sample width {sample_width} in {wav_path}")

    waveform = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    if num_channels > 1:
        waveform = waveform.reshape(-1, num_channels).mean(axis=1)

    waveform_tensor = torch.from_numpy(waveform)
    if sample_rate != target_sample_rate:
        waveform_tensor = F.interpolate(
            waveform_tensor.view(1, 1, -1),
            size=int(round(waveform_tensor.numel() * target_sample_rate / sample_rate)),
            mode="linear",
            align_corners=False,
        ).view(-1)

    if waveform_tensor.numel() < target_num_samples:
        waveform_tensor = F.pad(waveform_tensor, (0, target_num_samples - waveform_tensor.numel()))
    else:
        waveform_tensor = waveform_tensor[:target_num_samples]

    return waveform_tensor


def waveform_to_spectrogram(waveform, n_fft, hop_length, win_length):
    window = torch.hann_window(win_length)
    stft = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=True,
    )
    spectrogram = stft.abs()
    spectrogram = torch.log1p(spectrogram)
    return spectrogram.unsqueeze(0)


class SpeechCommandsDataset(Dataset):
    def __init__(self, samples, sample_rate, clip_duration_ms, n_fft, hop_length, win_length):
        self.samples = list(samples)
        self.sample_rate = sample_rate
        self.target_num_samples = int(sample_rate * clip_duration_ms / 1000)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        wav_path, label = self.samples[idx]
        waveform = read_waveform(wav_path, self.sample_rate, self.target_num_samples)
        spectrogram = waveform_to_spectrogram(waveform, self.n_fft, self.hop_length, self.win_length)
        return spectrogram, int(label)


def build_loader_kwargs(args, shuffle):
    use_cuda = torch.cuda.is_available()
    kwargs = {
        "shuffle": shuffle,
        "num_workers": args.num_workers,
        "pin_memory": use_cuda,
    }
    if args.num_workers > 0:
        kwargs["persistent_workers"] = True
    return kwargs


def dirichlet_partition(targets, num_clients, alpha, min_samples_per_client, seed):
    rng = np.random.default_rng(seed)
    targets = np.asarray(targets)
    classes = np.unique(targets)

    while True:
        client_indices = [[] for _ in range(num_clients)]
        for class_id in classes:
            class_indices = np.where(targets == class_id)[0]
            rng.shuffle(class_indices)

            proportions = rng.dirichlet(np.full(num_clients, alpha))
            split_points = (np.cumsum(proportions)[:-1] * len(class_indices)).astype(int)
            splits = np.split(class_indices, split_points)
            for client_id, split in enumerate(splits):
                client_indices[client_id].extend(split.tolist())

        client_sizes = [len(indices) for indices in client_indices]
        if min(client_sizes) >= min_samples_per_client:
            break

    for indices in client_indices:
        rng.shuffle(indices)
    return client_indices


def build_client_loaders(train_dataset, client_indices, batch_size, loader_kwargs):
    client_loaders = []
    client_sizes = []
    for indices in client_indices:
        subset = Subset(train_dataset, indices)
        client_loaders.append(DataLoader(subset, batch_size=batch_size, **loader_kwargs))
        client_sizes.append(len(indices))
    return client_loaders, client_sizes


def build_model():
    return AudioCNN(num_classes=len(COMMAND_LABELS))


def clone_state_dict_to_cpu(state_dict):
    return OrderedDict((key, value.detach().clone().cpu()) for key, value in state_dict.items())


def train_local_model(global_state, data_loader, sample_count, device, args, local_seed):
    model = build_model().to(device)
    model.load_state_dict(global_state)
    model.train()

    torch.manual_seed(local_seed)
    torch.cuda.manual_seed_all(local_seed)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.local_lr, weight_decay=args.weight_decay)
    for _ in range(args.local_epochs):
        for inputs, targets in data_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(inputs)
            loss = F.cross_entropy(logits, targets)
            loss.backward()
            optimizer.step()

    return clone_state_dict_to_cpu(model.state_dict()), sample_count


def aggregate_states(state_dicts, weights):
    total_weight = float(sum(weights))
    aggregated = OrderedDict()
    for key in state_dicts[0].keys():
        reference = state_dicts[0][key]
        if reference.is_floating_point():
            value = torch.zeros_like(reference)
            for state_dict, weight in zip(state_dicts, weights):
                value = value + state_dict[key] * (weight / total_weight)
            aggregated[key] = value
        else:
            aggregated[key] = reference.clone()
    return aggregated


def evaluate(model, data_loader, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(inputs)
            total_loss += F.cross_entropy(logits, targets, reduction="sum").item()
            total_correct += (logits.argmax(dim=1) == targets).sum().item()
            total_samples += targets.size(0)

    return total_loss / total_samples, total_correct / total_samples


def print_first_round_distribution(targets, selected_clients, client_indices):
    print("First-round client data distribution:")
    for client_id in selected_clients:
        client_targets = np.asarray(targets[client_indices[client_id]], dtype=np.int64)
        class_counts = np.bincount(client_targets, minlength=len(COMMAND_LABELS))
        total_samples = int(class_counts.sum())
        print(f"  Client {client_id}: total_samples={total_samples}")
        for class_id, count in enumerate(class_counts):
            ratio = (count / total_samples * 100.0) if total_samples > 0 else 0.0
            print(f"    class {class_id} ({COMMAND_LABELS[class_id]}): count={int(count)} ratio={ratio:.2f}%")


def run_federated_learning(args):
    if args.clients_per_round > args.num_clients:
        raise ValueError("clients_per_round must be less than or equal to num_clients")

    set_seed(args.seed)
    device = resolve_device(args)
    dataset_root = maybe_download_speech_commands(args.data_dir, args.download)

    train_samples = collect_samples(dataset_root, split="train", max_samples=args.max_train_samples)
    test_samples = collect_samples(dataset_root, split="test", max_samples=args.max_test_samples)
    if not train_samples or not test_samples:
        raise RuntimeError("Failed to collect Speech Commands samples.")

    train_targets = np.asarray([label for _, label in train_samples], dtype=np.int64)
    train_dataset = SpeechCommandsDataset(
        train_samples,
        sample_rate=args.sample_rate,
        clip_duration_ms=args.clip_duration_ms,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length,
    )
    test_dataset = SpeechCommandsDataset(
        test_samples,
        sample_rate=args.sample_rate,
        clip_duration_ms=args.clip_duration_ms,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length,
    )

    client_indices = dirichlet_partition(
        train_targets,
        args.num_clients,
        args.dirichlet_alpha,
        args.min_samples_per_client,
        args.seed,
    )

    train_loader_kwargs = build_loader_kwargs(args, shuffle=True)
    eval_loader_kwargs = build_loader_kwargs(args, shuffle=False)
    client_loaders, client_sizes = build_client_loaders(
        train_dataset,
        client_indices,
        args.batch_size,
        train_loader_kwargs,
    )
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, **eval_loader_kwargs)

    print(
        "Dataset setup:",
        f"dataset=Speech Commands",
        f"classes={len(COMMAND_LABELS)}",
        f"labels={COMMAND_LABELS}",
        f"train_samples={len(train_samples)}",
        f"test_samples={len(test_samples)}",
    )
    print(
        "Federated setup:",
        f"clients={args.num_clients}",
        f"sampled_per_round={args.clients_per_round}",
        f"rounds={args.rounds}",
        f"local_epochs={args.local_epochs}",
        f"dirichlet_alpha={args.dirichlet_alpha}",
        f"device={device}",
    )
    print(
        "Client data sizes:",
        f"min={min(client_sizes)} max={max(client_sizes)} avg={sum(client_sizes) / len(client_sizes):.1f}",
    )

    global_model = build_model().to(device)

    for round_idx in range(1, args.rounds + 1):
        selected_clients = random.sample(range(args.num_clients), args.clients_per_round)
        if round_idx == 1:
            print_first_round_distribution(train_targets, selected_clients, client_indices)

        global_state = clone_state_dict_to_cpu(global_model.state_dict())
        local_states = []
        sample_counts = []

        for client_id in selected_clients:
            local_seed = args.seed + round_idx * 1000 + client_id
            local_state, sample_count = train_local_model(
                global_state,
                client_loaders[client_id],
                client_sizes[client_id],
                device,
                args,
                local_seed,
            )
            local_states.append(local_state)
            sample_counts.append(sample_count)

        next_state = aggregate_states(local_states, sample_counts)
        global_model.load_state_dict(next_state)

        test_loss, test_acc = evaluate(global_model, test_loader, device)
        print(
            f"Round {round_idx:03d}:",
            f"selected_clients={selected_clients}",
            f"test_loss={test_loss:.4f}",
            f"test_acc={test_acc:.4f}",
        )

    return global_model


def main():
    args = parse_args()
    run_federated_learning(args)


if __name__ == "__main__":
    main()
