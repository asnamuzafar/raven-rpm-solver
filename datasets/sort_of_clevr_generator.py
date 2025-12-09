"""
Sort-of-CLEVR Dataset Generator
================================
A simplified version of CLEVR for visual relational reasoning.

Dataset Structure:
- Images: 128x128 with 6 colored shapes
- Questions: 
  - Non-relational: "What color is the [shape]?"
  - Relational: "What is the shape of the object closest to the [color] object?"

Expected Results:
- CNN+MLP: ~60% relational, ~95% non-relational
- Relation Network: ~95% relational, ~95% non-relational
"""

import os
import random
import pickle
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm


# Colors: (R, G, B, name)
COLORS = [
    ((255, 0, 0), 'red'),
    ((0, 255, 0), 'green'),
    ((0, 0, 255), 'blue'),
    ((255, 255, 0), 'yellow'),
    ((255, 165, 0), 'orange'),
    ((128, 128, 128), 'gray'),
]

# Shapes
SHAPES = ['circle', 'rectangle']


def draw_shape(draw, x, y, size, shape, color):
    """Draw a shape on the image."""
    half = size // 2
    if shape == 'circle':
        draw.ellipse([x-half, y-half, x+half, y+half], fill=color)
    else:  # rectangle
        draw.rectangle([x-half, y-half, x+half, y+half], fill=color)


def generate_image(img_size=128, num_objects=6, min_size=10, max_size=15):
    """
    Generate a Sort-of-CLEVR image.
    
    Returns:
        img: PIL Image
        objects: List of (x, y, color_idx, shape_idx) for each object
    """
    img = Image.new('RGB', (img_size, img_size), color='black')
    draw = ImageDraw.Draw(img)
    
    objects = []
    positions = []
    
    for i in range(num_objects):
        # Pick color (each object has unique color)
        color_idx = i
        color = COLORS[color_idx][0]
        
        # Pick shape randomly
        shape_idx = random.randint(0, len(SHAPES)-1)
        shape = SHAPES[shape_idx]
        
        # Pick size
        size = random.randint(min_size, max_size)
        
        # Find non-overlapping position
        margin = max_size + 5
        for _ in range(100):
            x = random.randint(margin, img_size - margin)
            y = random.randint(margin, img_size - margin)
            
            # Check distance from other objects
            overlap = False
            for px, py in positions:
                if abs(x - px) < margin and abs(y - py) < margin:
                    overlap = True
                    break
            
            if not overlap:
                break
        
        positions.append((x, y))
        objects.append((x, y, color_idx, shape_idx, size))
        draw_shape(draw, x, y, size, shape, color)
    
    return img, objects


def generate_question_answer(objects):
    """
    Generate a question-answer pair.
    
    Question types:
    - Non-relational: "What is the shape of the [color] object?"
    - Relational: "What shape is the object closest to the [color] object?"
    
    Returns:
        question: one-hot encoded question
        answer: class index
        q_type: 'relational' or 'non-relational'
    """
    num_objects = len(objects)
    
    # Choose question type
    q_type = random.choice(['relational', 'non-relational'])
    
    # Choose target object (by color)
    target_idx = random.randint(0, num_objects - 1)
    target = objects[target_idx]
    
    if q_type == 'non-relational':
        # "What is the shape of the [color] object?"
        # Answer: shape index
        answer = target[3]  # shape_idx
        
        # Encode question: [color_one_hot (6)] + [q_type (2)]
        question = np.zeros(8, dtype=np.float32)
        question[target_idx] = 1  # color
        question[6] = 1  # non-relational
        
    else:  # relational
        # "What shape is the object closest to the [color] object?"
        tx, ty = target[0], target[1]
        
        # Find closest object
        min_dist = float('inf')
        closest_idx = -1
        for i, obj in enumerate(objects):
            if i != target_idx:
                dist = (obj[0] - tx)**2 + (obj[1] - ty)**2
                if dist < min_dist:
                    min_dist = dist
                    closest_idx = i
        
        answer = objects[closest_idx][3]  # shape of closest
        
        # Encode question
        question = np.zeros(8, dtype=np.float32)
        question[target_idx] = 1  # color
        question[7] = 1  # relational
    
    return question, answer, q_type


def generate_dataset(num_images=10000, questions_per_image=10, img_size=128, save_dir='./data/sort_of_clevr'):
    """
    Generate complete Sort-of-CLEVR dataset.
    
    Args:
        num_images: Number of images to generate
        questions_per_image: Questions per image
        img_size: Image size
        save_dir: Directory to save dataset
    """
    os.makedirs(save_dir, exist_ok=True)
    
    data = {
        'train': {'images': [], 'questions': [], 'answers': [], 'types': []},
        'val': {'images': [], 'questions': [], 'answers': [], 'types': []},
        'test': {'images': [], 'questions': [], 'answers': [], 'types': []},
    }
    
    # Split: 70% train, 15% val, 15% test
    train_size = int(num_images * 0.7)
    val_size = int(num_images * 0.15)
    
    print(f"Generating {num_images} images with {questions_per_image} questions each...")
    
    for i in tqdm(range(num_images)):
        # Determine split
        if i < train_size:
            split = 'train'
        elif i < train_size + val_size:
            split = 'val'
        else:
            split = 'test'
        
        # Generate image
        img, objects = generate_image(img_size)
        img_array = np.array(img).transpose(2, 0, 1).astype(np.float32) / 255.0
        
        # Generate questions
        for _ in range(questions_per_image):
            question, answer, q_type = generate_question_answer(objects)
            
            data[split]['images'].append(img_array)
            data[split]['questions'].append(question)
            data[split]['answers'].append(answer)
            data[split]['types'].append(0 if q_type == 'non-relational' else 1)
    
    # Convert to numpy
    for split in data:
        for key in data[split]:
            data[split][key] = np.array(data[split][key])
        
        print(f"{split}: {len(data[split]['images'])} samples")
    
    # Save
    for split, split_data in data.items():
        filepath = os.path.join(save_dir, f'{split}.pkl')
        with open(filepath, 'wb') as f:
            pickle.dump(split_data, f)
        print(f"Saved {filepath}")
    
    return data


class SortOfCLEVRDataset:
    """PyTorch-compatible dataset for Sort-of-CLEVR."""
    
    def __init__(self, data_dir, split='train', relational_only=False, non_relational_only=False):
        filepath = os.path.join(data_dir, f'{split}.pkl')
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.images = data['images']
        self.questions = data['questions']
        self.answers = data['answers']
        self.types = data['types']
        
        # Filter by question type if needed
        if relational_only:
            mask = self.types == 1
            self.images = self.images[mask]
            self.questions = self.questions[mask]
            self.answers = self.answers[mask]
            self.types = self.types[mask]
        elif non_relational_only:
            mask = self.types == 0
            self.images = self.images[mask]
            self.questions = self.questions[mask]
            self.answers = self.answers[mask]
            self.types = self.types[mask]
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        import torch
        return (
            torch.from_numpy(self.images[idx]),
            torch.from_numpy(self.questions[idx]),
            torch.tensor(self.answers[idx], dtype=torch.long),
            torch.tensor(self.types[idx], dtype=torch.long)
        )


if __name__ == '__main__':
    import argparse
    
    # Get the parent directory of the script location (for consistent data path)
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PARENT_DIR = os.path.dirname(SCRIPT_DIR)
    DEFAULT_SAVE_DIR = os.path.join(PARENT_DIR, 'data', 'sort_of_clevr')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_images', type=int, default=10000)
    parser.add_argument('--questions_per_image', type=int, default=10)
    parser.add_argument('--save_dir', type=str, default=DEFAULT_SAVE_DIR)
    args = parser.parse_args()
    
    generate_dataset(
        num_images=args.num_images,
        questions_per_image=args.questions_per_image,
        save_dir=args.save_dir
    )
    
    print("\n✓ Dataset generation complete!")
