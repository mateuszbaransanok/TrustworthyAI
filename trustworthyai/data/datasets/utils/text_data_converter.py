import shutil
from collections import defaultdict

import pandas as pd

from trustworthyai.settings import DATA_DIR


class TextDataConverter:
    ROOT_DIR = DATA_DIR.joinpath('datasets', 'text')

    def load(
        self,
        name: str,
    ) -> pd.DataFrame:
        records = []
        name, _, version = name.partition('/')

        for split in ('train', 'val', 'test', 'predict'):
            split_dir = self.ROOT_DIR.joinpath(name, 'annotations', version, split + '.txt')

            if not split_dir.exists():
                continue

            for line in split_dir.read_text().splitlines():
                rel_path, is_label, label = line.partition(' ')
                path = self.ROOT_DIR.joinpath(name, rel_path)
                text = self.ROOT_DIR.joinpath(name, path).read_text()
                record = {
                    'index': int(path.stem),
                    **({'category': label} if is_label else {}),
                    'split': split,
                    'text': text,
                }
                records.append(record)

        return pd.DataFrame.from_records(records)

    def save(
        self,
        df: pd.DataFrame,
        name: str,
        force: bool = False,
    ) -> None:
        name, _, version = name.partition('/')
        texts_dir = self.ROOT_DIR.joinpath(name, 'texts')
        if not version:
            if texts_dir.exists():
                shutil.rmtree(texts_dir)
            texts_dir.mkdir(parents=True)

        annotations_dir = self.ROOT_DIR.joinpath(name, 'annotations', version)
        if annotations_dir.exists():
            shutil.rmtree(annotations_dir)
        annotations_dir.mkdir(parents=True)

        splits = defaultdict(list)
        categories = set()

        for _, row in df.iterrows():
            index = row['index']
            split = row['split']
            text = row['text']

            batch_directory = texts_dir.joinpath(str(index // 1000))
            path = batch_directory.joinpath(str(index)).with_suffix('.txt')

            if not version or force:
                batch_directory.mkdir(parents=True, exist_ok=True)
                path.write_text(text)

            if 'category' in row:
                category = row['category']
                categories.add(category)
                line = f'{path.relative_to(self.ROOT_DIR / name)} {category}'
            else:
                line = f'{path.relative_to(self.ROOT_DIR / name)}'

            splits[split].append(line)

        for split, lines in splits.items():
            text = '\n'.join(lines) + '\n'
            annotations_dir.joinpath(split).with_suffix('.txt').write_text(text)

        if categories:
            annotations_dir.joinpath('classes.txt').write_text('\n'.join(sorted(categories)) + '\n')
