import grain.python as pygrain
from pathlib import Path
import tiktoken


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
        batch_size: Batch size for training
        maxlen: Maximum sequence length
        max_stories: Maximum number of stories to load (for memory efficiency)
        num_epochs: Number of training epochs
        shuffle: Whether to shuffle the data
        seed: Random seed for reproducibility

    Returns:
        Tuple of (Grain DataLoader, estimated_batches_per_epoch)
    """

    # Load and validate file
    file_path = file_path

    print(f"Loading data from {file_path} (max {max_stories:,} stories)")

    # Read file in chunks to avoid loading entire file into memory
    stories = []
    current_story = []

    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            if '<|endoftext|>' in line:
                # Split on end token and process parts
                parts = line.split('<|endoftext|>')
                for i, part in enumerate(parts[:-1]):  # All but last part have end tokens
                    current_story.append(part)
                    story_text = ''.join(current_story).strip()
                    if story_text:
                        stories.append(story_text + '<|endoftext|>')
                        if len(stories) >= max_stories:
                            break
                    current_story = []

                # Last part becomes start of next story
                if parts[-1].strip():
                    current_story = [parts[-1]]

                if len(stories) >= max_stories:
                    break
            else:
                current_story.append(line)

        # Don't forget the last story if file doesn't end with end token
        if current_story and len(stories) < max_stories:
            story_text = ''.join(current_story).strip()
            if story_text:
                stories.append(story_text + '<|endoftext|>')

    print(f"Loaded {len(stories):,} stories")
    if len(stories) == 0:
        raise ValueError("No valid stories found in the dataset")

    # Calculate estimated batches per epoch
    estimated_batches_per_epoch = len(stories) // batch_size
    print(f"Estimated batches per epoch: {estimated_batches_per_epoch:,}")

    # Create efficient dataset
    dataset = StoryDataset(stories, maxlen, tokenizer)

    # Configure sampler with sharding support
    sampler = pygrain.IndexSampler(
        num_records=len(dataset),
        shuffle=shuffle,
        seed=seed,
        shard_options=pygrain.NoSharding(),
        num_epochs=num_epochs,
    )

    # Create DataLoader with efficient batching
    dataloader = pygrain.DataLoader(
        data_source=dataset,
        sampler=sampler,
        operations=[
            pygrain.Batch(batch_size=batch_size, drop_remainder=True)
        ]
    )

    print(f"Created DataLoader with batch_size={batch_size}, maxlen={maxlen}")
    return dataloader, estimated_batches_per_epoch