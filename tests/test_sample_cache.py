import numpy as np
from unittest.mock import patch
from my_predictor.data import ManifestDataset, pad_batch


def test_cache_loads_once_and_preserves_padded_samples():
    sample=np.array([[1.,2.],[3.,4.]],dtype=np.float32)
    rows=[dict(id="example",labels="unused")]
    with patch("my_predictor.data.read_manifest",return_value=rows), patch("my_predictor.data.load_manifest_embedding",return_value=sample) as load, patch("my_predictor.data.load_vector",return_value=np.array([1,2])):
        cached=ManifestDataset("unused",cache_samples=True)
        one=cached[0]
        before=one["embedding"].copy()
        first=pad_batch([one])
        again=pad_batch([cached[0]])
        assert load.call_count==1
        np.testing.assert_array_equal(before,one["embedding"])
        for a,b in zip(first[:4],again[:4]):
            if isinstance(a,list):assert a==b
            else:assert a.equal(b)
        assert one["valid"].tolist()==[True,False]


def test_default_still_reads_each_time():
    with patch("my_predictor.data.read_manifest",return_value=[dict(id="x",labels="unused")]), patch("my_predictor.data.load_manifest_embedding",return_value=np.zeros((1,2),dtype=np.float32)) as load, patch("my_predictor.data.load_vector",return_value=np.array([0])):
        dataset=ManifestDataset("unused")
        dataset[0];dataset[0]
        assert load.call_count==2
