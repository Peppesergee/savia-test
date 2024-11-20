package cineca.AL_LLM;

import org.apache.pdfbox.multipdf.Splitter;
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.text.PDFTextStripper;
import org.json.JSONArray;
import org.json.JSONObject;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.List;

public class PdfReader {

    @SuppressWarnings({"all"})
    public static void readPDF(String baseFolder, String outFile) throws IOException {

//        String baseFolder = "../datasets/downloaded/";
        File f = new File(baseFolder);

        // Populates the array with names of files and directories
        String[] pathnames = f.list();
        Arrays.sort(pathnames);

        JSONArray js_arr = new JSONArray();
        System.out.println("Number of files: " + pathnames.length);

//        for (int item = 10; item < 18; item++) {
//            String pathname = pathnames[item];

        int failed = 0;

        for (String pathname: pathnames){

            System.out.println(pathname);
            File file = new File(baseFolder + "/" + pathname);

            try {
                PDDocument document = PDDocument.load(file);
                PDFTextStripper stripper = new PDFTextStripper();

                Splitter splitter = new Splitter();
                List<PDDocument> pages = splitter.split(document);

                JSONObject json_obj_doc = new JSONObject();
                JSONArray json_arr_pages = new JSONArray();

                for (int i = 0; i < pages.size(); i++) {
                    PDDocument doc = pages.get(i);
                    String page_text = stripper.getText(doc);
                    json_arr_pages.put(page_text);
                }

                json_obj_doc.put("filename", pathname);
                json_obj_doc.put("pages", json_arr_pages);

                js_arr.put(json_obj_doc);

                document.close();
            }
            catch (IOException e){
                System.out.println("unable to open file " + pathname);
                failed += 1;
            }

        }

        try (PrintWriter out = new PrintWriter(outFile)) {
            out.println(js_arr);
        }

        System.out.println("Failed to load " +  String.valueOf(failed) + " files");

    }

}
