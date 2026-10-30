"""
Loading model weights files, and the message given when a file cannot be read.

A weights file left incomplete by an interrupted download, or an error page saved in
place of one, must fail with a message that names the file and says how to replace it.
Other failures keep PyTorch's own error, so a real problem is not mislabelled.
"""

import zipfile

import pytest
import torch

from trapdata.ml.utils import load_model_checkpoint


@pytest.fixture
def checkpoint_path(tmp_path):
    path = tmp_path / "weights.pth"
    torch.save({"model_state_dict": {"weight": torch.zeros(1000)}}, path)
    return path


def test_intact_checkpoint_loads(checkpoint_path):
    checkpoint = load_model_checkpoint(checkpoint_path, "cpu")

    assert torch.equal(checkpoint["model_state_dict"]["weight"], torch.zeros(1000))


@pytest.mark.parametrize("keep_fraction", [0.5, 0.99], ids=["half", "nearly-all"])
def test_truncated_checkpoint_names_the_file_and_the_fix(
    checkpoint_path, keep_fraction
):
    """
    A partly downloaded file fails with a message that names the file and says to
    delete it. PyTorch's own error for this case names neither.
    """
    data = checkpoint_path.read_bytes()
    checkpoint_path.write_bytes(data[: int(len(data) * keep_fraction)])

    with pytest.raises(RuntimeError, match="incomplete or corrupted") as excinfo:
        load_model_checkpoint(checkpoint_path, "cpu")

    message = str(excinfo.value)
    assert str(checkpoint_path) in message
    assert "Delete it" in message


def test_error_page_saved_as_weights_names_the_file(checkpoint_path):
    checkpoint_path.write_text("<html><body>404 Not Found</body></html>")

    with pytest.raises(RuntimeError, match="incomplete or corrupted") as excinfo:
        load_model_checkpoint(checkpoint_path, "cpu")

    assert str(checkpoint_path) in str(excinfo.value)


def test_missing_file_is_reported_as_missing(tmp_path):
    """A file that is not there at all is reported as missing, not as corrupted."""
    with pytest.raises(FileNotFoundError):
        load_model_checkpoint(tmp_path / "absent.pth", "cpu")


def test_intact_archive_that_fails_to_load_keeps_the_original_error(tmp_path):
    """
    Only a file that is not a whole zip archive is called incomplete. A complete
    archive that still fails to load gets PyTorch's own error, so the message does
    not send someone to re-download a file that downloaded correctly.
    """
    path = tmp_path / "not-a-checkpoint.pth"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("readme.txt", "not a checkpoint")

    with pytest.raises(RuntimeError) as excinfo:
        load_model_checkpoint(path, "cpu")

    assert "incomplete or corrupted" not in str(excinfo.value)
