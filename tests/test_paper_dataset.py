import csv
import json
import numpy as np
import pytest
from my_predictor.paper_dataset import prepare_training_inputs
from my_predictor.io import read_manifest
from my_predictor.data import ManifestDataset


def inputs(tmp_path, corrupt=False):
    fasta = tmp_path/'sequences.fasta'
    fasta.write_text('>train-protein\nAX\n>val-protein\nCY\n')
    records = tmp_path/'development.jsonl'
    rows = [{'id':'train-protein','sequence':'AX','split':'train','labels':[1,2],'mask':[1,1]},
            {'id':'val-protein','sequence':'CY' if not corrupt else 'CC','split':'val','labels':[0,1],'mask':[1,0]}]
    records.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    for family,width,key in [('prott5',1024,'layer_24'),('esmc',1152,'layer_36')]:
        array = tmp_path/(family+'.npz')
        np.savez(array, **{key:np.zeros((2,width),dtype=np.float32)})
        with (tmp_path/(family+'.tsv')).open('w') as f:
            w=csv.DictWriter(f,fieldnames=['id','path','key','family'],delimiter='\t'); w.writeheader()
            for identity in ['train-protein','val-protein']:
                w.writerow(dict(id=identity,path=array.name,key=key,family=family))
    return fasta,tmp_path/'prott5.tsv',tmp_path/'esmc.tsv',records


def test_training_preparation_preserves_splits_unknowns_and_masks(tmp_path):
    args=inputs(tmp_path)
    output=prepare_training_inputs(*args,tmp_path/'prepared')
    train=ManifestDataset(output/'prott5-train.tsv')[0]
    val=ManifestDataset(output/'esmc-val.tsv')[0]
    assert train['embedding'].shape == (2,1066)
    assert train['valid'].tolist() == [True,False]
    assert val['embedding'].shape == (2,1194)
    assert val['valid'].tolist() == [True,False]
    assert 'labels' not in read_manifest(output/'features/prott5.tsv')[0]
    assert read_manifest(output/'prott5-train.tsv')[0]['split']=='train'


def test_training_preparation_rejects_sequence_mismatch_before_output(tmp_path):
    args=inputs(tmp_path,corrupt=True)
    with pytest.raises(ValueError,match='differs from FASTA'):
        prepare_training_inputs(*args,tmp_path/'bad')
    assert not (tmp_path/'bad').exists()
