### PdfReader

Tool per l'estrazione del testo dai pdf, basato su Apache PDFBox (https://pdfbox.apache.org/).


Uso del jar:

java -jar executable.jar -i ./cartella_di_input_con_i_pdf -o json_di_output.json

output:

[
    {

    "filename": "filename.pdf",

    "pages": [
        "testo di pagina 1", 
        "testo di pagina 2"
    ]
]