"""Attach frozen development labels to separately prepared paper features."""
from __future__ import annotations
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
from .io import read_fasta, read_manifest
from .paper_inference import prepare_inputs


def _label(value):
    if isinstance(value, bool) or str(value) not in ('0', '1', '2'):
        raise ValueError(f'labels must contain only 0/1/2, got {value!r}')
    return int(value)


def _mask(value):
    if value not in (True, False, 0, 1, '0', '1'):
        raise ValueError(f'mask values must be booleans or 0/1, got {value!r}')
    return value in (True, 1, '1')


def prepare_training_inputs(fasta, prott5_index, esmc_index, records_path, output_dir):
    fasta, records_path = Path(fasta), Path(records_path)
    sequences = {record.id: record.sequence for record in read_fasta(fasta)}
    if not sequences:
        raise ValueError('empty FASTA')
    records = {}
    opener = gzip.open if str(records_path).endswith('.gz') else open
    with opener(records_path, 'rt', encoding='utf-8') as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            identity = str(record.get('id') or record.get('accession') or record.get('record_key') or '')
            if not identity or identity in records:
                raise ValueError(f'blank or duplicate development ID: {identity!r}')
            split = record.get('split')
            if split not in ('train', 'val', 'test'):
                raise ValueError(f'{identity}: split must be train, val or test')
            sequence = record.get('sequence')
            if sequences.get(identity) != sequence:
                raise ValueError(f'{identity}: development sequence differs from FASTA')
            labels = np.asarray([_label(x) for x in record['labels']], dtype=np.int64)
            raw_mask = record.get('mask', record.get('label_mask'))
            mask = np.asarray([_mask(x) for x in raw_mask], dtype=bool) if raw_mask is not None else labels != 2
            if labels.shape != (len(sequence),) or mask.shape != labels.shape:
                raise ValueError(f'{identity}: labels/mask length differs from sequence')
            records[identity] = (split, labels, mask)
    if set(records) != set(sequences):
        raise ValueError('FASTA and development labels must contain exactly the same IDs')
    if not {'train', 'val'}.issubset({record[0] for record in records.values()}):
        raise ValueError('development labels must include train and val splits')
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f'refusing to overwrite existing output: {output}')
    # Labels and feature-only predictor inputs have distinct documented locations.
    output.mkdir(parents=True)
    features = prepare_inputs(fasta, prott5_index, esmc_index, output / 'features')
    labels_dir = output / 'labels'
    labels_dir.mkdir()
    paths = {}
    for identity, (_, labels, mask) in records.items():
        token = hashlib.sha256(identity.encode()).hexdigest()
        label_path, mask_path = labels_dir / (token + '.labels.npy'), labels_dir / (token + '.mask.npy')
        np.save(label_path, labels, allow_pickle=False)
        np.save(mask_path, mask, allow_pickle=False)
        paths[identity] = (str(label_path), str(mask_path))
    manifests = {}
    for family in ('prott5', 'esmc'):
        source_rows = read_manifest(features / (family + '.tsv'))
        for split in ('train', 'val', 'test'):
            rows = []
            for row in source_rows:
                identity = row['id']
                if records[identity][0] == split:
                    label_path, mask_path = paths[identity]
                    rows.append(dict(row, split=split, labels=label_path, mask=mask_path))
            if not rows:
                continue
            destination = output / f'{family}-{split}.tsv'
            with destination.open('x', encoding='utf-8', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter='\t', lineterminator='\n')
                writer.writeheader()
                writer.writerows(rows)
            manifests[f'{family}-{split}'] = destination.name
    receipt = {'schema_version': 1, 'label_source_sha256': hashlib.sha256(records_path.read_bytes()).hexdigest(),
               'fasta_sha256': hashlib.sha256(fasta.read_bytes()).hexdigest(), 'manifests': manifests,
               'split_counts': {split: sum(value[0] == split for value in records.values()) for split in ('train','val','test')},
               'policy': 'frozen supplied splits; no model selection or feature fitting during preparation'}
    (output / 'preparation.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return output


def prepare_training_main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fasta', required=True)
    parser.add_argument('--prott5-index', required=True)
    parser.add_argument('--esmc-index', required=True)
    parser.add_argument('--records', required=True, help='Frozen development JSONL(.gz): id,sequence,split,labels,mask')
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args(argv)
    print(prepare_training_inputs(args.fasta, args.prott5_index, args.esmc_index, args.records, args.output_dir))
