"""Test TransferableObject model."""

import pytest

from pps.io.transfer.model import TransferableObject


class TestTransferableObject:
    def test_repr(self):
        """Test the __repr__ method."""
        obj = TransferableObject(source='source_path', destination='destination_path')
        assert repr(obj) == 'TransferableObject(source=source_path, destination=destination_path)'

    def test_str(self):
        """Test the __str__ method."""
        obj = TransferableObject(source='source_path', destination='destination_path')
        assert str(obj) == 'TransferableObject(source=source_path, destination=destination_path)'

    def test_transfer(self):
        """Test the transfer method raises NotImplementedError."""
        obj = TransferableObject(source='source_path', destination='destination_path')
        with pytest.raises(NotImplementedError, match='Implement in derivative class'):
            obj.transfer()
