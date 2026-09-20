import grain.python as pygrain
from pathlib import Path


class StoryDataset:
    def __init__(self, stories, maxlen, tokenizer):
        self.stories = stories
        self.maxlen = maxlen
        self.tokenizer = tokenizer
        self.end_token = tokenizer.encode(
            '<|endoftext|>',
            allowed_special={'<|endoftext|>'}
        )[0]

    def __len__(self):
        return len(self.stories)

    def __getitem__(self, idx):
        story = self.stories[idx]
        tokens = self.tokenizer.encode(
            story,
            allowed_special={'<|endoftext|>'}
        )

        if len(tokens) > self.maxlen:
            tokens = tokens[:self.maxlen]

        seq_len = len(tokens)

        tokens.extend([0] * (self.maxlen - len(tokens)))
        return {
            "tokens": tokens,
            "seq_len": seq_len
        }


def load_stories_from_file(
    file_path,
    max_stories = None
):
    """
    Efficiently load stories from a text file.
    Each story ends with <|endoftext|>.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    print(f"Loading stories from {file_path}...")
    stories = []
    current_story = []

    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            if '<|endoftext|>' in line:
                parts = line.split('<|endoftext|>')
                for part in parts[:-1]:
                    current_story.append(part)
                    story_text = ''.join(current_story).strip()
                    if story_text:
                        stories.append(story_text + '<|endoftext|>')
                        if max_stories and len(stories) >= max_stories:
                            break
                    current_story = []
                if parts[-1].strip():
                    current_story = [parts[-1]]
                else:
                    current_story = []
                if max_stories and len(stories) >= max_stories:
                    break
            else:
                current_story.append(line)
        if current_story and (not max_stories or len(stories) < max_stories):
            story_text = ''.join(current_story).strip()
            if story_text:
                stories.append(story_text + '<|endoftext|>')

    print(f"Loaded {len(stories):,} stories")
    return stories


def load_and_preprocess_data(
    file_path,
    tokenizer,
    batch_size,
    maxlen,
    max_stories = 100_000,
    num_epochs = 1,
    shuffle = False,
    seed = 42
):
    """
    Load and preprocess TinyStories data with memory-efficient chunk reading.

    Args:
        file_path: Path to the text file
        tokenizer: Tokenizer to use
        batch_size: Batch size for training
        maxlen: Maximum sequence length
        max_stories: Maximum number of stories to load (for memory efficiency)
        num_epochs: Number of training epochs
        shuffle: Whether to shuffle the data
        seed: Random seed for reproducibility

    Returns:
        Tuple of (Grain DataLoader, estimated_batches_per_epoch)
    """

    stories = load_stories_from_file(file_path, max_stories)
    loader = create_dataloader(
        stories, tokenizer, batch_size, maxlen,
        num_epochs=num_epochs, shuffle=shuffle, seed=seed,
    )
    return loader, len(stories) // batch_size


def create_dataloader(
    stories, tokenizer, batch_size, maxlen, *,
    num_epochs=1, shuffle=False, seed=42, drop_remainder=True,
):
    """Create a loader; retain partial batches when evaluating validation data."""
    if not stories:
        raise ValueError("No valid stories found in the dataset")
    dataset = StoryDataset(stories, maxlen, tokenizer)
    sampler = pygrain.IndexSampler(
        num_records=len(dataset),
        shuffle=shuffle,
        seed=seed,
        shard_options=pygrain.NoSharding(),
        num_epochs=num_epochs,
    )
    return pygrain.DataLoader(
        data_source=dataset,
        sampler=sampler,
        operations=[pygrain.Batch(batch_size=batch_size, drop_remainder=drop_remainder)],
    )


def split_stories(stories, validation_fraction=0.1, seed=42):
    """Split whole stories reproducibly, independently of the model/training seed."""
    import random

    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    # Keep identical story texts together so duplicates cannot leak across splits.
    unique_stories = list(dict.fromkeys(stories))
    if len(unique_stories) < 2:
        raise ValueError("Need at least two distinct stories for train/validation")
    random.Random(seed).shuffle(unique_stories)
    count = min(len(unique_stories) - 1, max(1, int(len(unique_stories) * validation_fraction)))
    return unique_stories[count:], unique_stories[:count]
