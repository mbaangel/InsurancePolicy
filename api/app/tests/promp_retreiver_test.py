import os
import pytest

from app.model.pdf_text import text_extract, extract_text_from_pdf # Assuming extract_text_from_pdf is defined elsewhere

# Create dummy PDF files for testing
TEST_PDF_FOLDER = "./test_pdfs"

def setup_module():
    os.makedirs(TEST_PDF_FOLDER, exist_ok=True)
    with open(os.path.join(TEST_PDF_FOLDER, "test1.pdf"), "w") as f:
        f.write("This is a test PDF.")
    with open(os.path.join(TEST_PDF_FOLDER, "test2.pdf"), "w") as f:
        f.write("This is another test PDF.")

def teardown_module():
    # Clean up test files
    for file in os.listdir(TEST_PDF_FOLDER):
        os.remove(os.path.join(TEST_PDF_FOLDER, file))
    os.rmdir(TEST_PDF_FOLDER)


@pytest.mark.parametrize(
    "pdf_folder, expected_files",
    [
        (TEST_PDF_FOLDER, ["test1.pdf", "test2.pdf"]), # happy path: multiple PDFs
        ("./docs/dataset", []), # edge case: empty directory
    ],
    ids=["multiple_pdfs", "empty_directory"]
)
def test_text_extract_happy_path(pdf_folder, expected_files, monkeypatch):

    # Arrange
    def mock_extract_text_from_pdf(pdf_path):
        return "Mock text from " + os.path.basename(pdf_path)

    monkeypatch.setattr("app.model.pdf_text.extract_text_from_pdf", mock_extract_text_from_pdf)

    # Act
    pdf_texts = text_extract(pdf_folder)

    # Assert
    assert len(pdf_texts) == len(expected_files)
    for file in expected_files:
        assert file in pdf_texts
        assert pdf_texts[file] == "Mock text from " + file


def test_text_extract_no_pdfs(monkeypatch, tmp_path):
    # Arrange
    pdf_folder = tmp_path / "pdfs"
    pdf_folder.mkdir()
    (pdf_folder / "test.txt").write_text("This is a text file.")

    def mock_extract_text_from_pdf(pdf_path):
        return "Mock text"

    monkeypatch.setattr("app.model.pdf_text.extract_text_from_pdf", mock_extract_text_from_pdf)

    # Act
    pdf_texts = text_extract(str(pdf_folder))

    # Assert
    assert len(pdf_texts) == 0


def test_text_extract_exception(monkeypatch, tmp_path):
    # Arrange
    pdf_folder = tmp_path / "pdfs"
    pdf_folder.mkdir()
    (pdf_folder / "test.pdf").write_text("This is a test PDF.")

    def mock_extract_text_from_pdf(pdf_path):
        raise Exception("Mock exception")

    monkeypatch.setattr("app.model.pdf_text.extract_text_from_pdf", mock_extract_text_from_pdf)

    # Act and Assert
    with pytest.raises(Exception, match="Mock exception"):
        text_extract(str(pdf_folder))

