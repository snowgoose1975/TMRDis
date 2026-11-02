import numpy as np
import pytest
from my_predictor.feature_preprocessing import apply_preprocessing

def test_mixed_unit_raw_and_original_norm():
    x=np.array([[3.,4.,2.,-6.],[0.,0.,1.,2.]],dtype=np.float32)
    metadata=dict(method="unit",segments=[2,2],segment_methods=["unit","raw"],append_norm=True)
    y=apply_preprocessing(x,metadata)
    np.testing.assert_allclose(y[:,:2], [[.6,.8],[0,0]])
    np.testing.assert_array_equal(y[:,2:4],x[:,2:4])
    np.testing.assert_allclose(y[:,4:],np.log1p(np.column_stack([np.linalg.norm(x[:,:2],axis=1),np.linalg.norm(x[:,2:],axis=1)])))

def test_invalid_segment_count():
    with pytest.raises(ValueError):
        apply_preprocessing(np.ones((1,4)),dict(method="unit",segments=[2,2],segment_methods=["unit"]))
