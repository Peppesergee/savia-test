package cineca.AL_LLM;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

import org.apache.pdfbox.multipdf.Splitter;
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.text.PDFTextStripper;
import org.json.JSONArray;
import org.json.JSONObject;

import static cineca.AL_LLM.PdfReader.readPDF;
import org.apache.commons.cli.*;


public class Main {
    public static void main(String[] args) {

        Options options = new Options();
//        Option alpha = new Option("i", "input-folder", false, "Activate feature alpha", "argName": "input-folder" );

        Option inputFolder = Option.builder("i").longOpt("input-folder").argName("input-folder").hasArg().required(true).desc("folder containing pdfs").build();
        options.addOption(inputFolder);
        Option outputFile = Option.builder("o").longOpt("output-file").argName("output-file").hasArg().required(true).desc("output file (JSON)").build();
        options.addOption(outputFile);

        CommandLine cmd;
        CommandLineParser parser = new BasicParser();
        HelpFormatter helper = new HelpFormatter();

        String baseFolder = "";
        String outFile = "";

        try {
            cmd = parser.parse(options, args);
            if (cmd.hasOption("i")) {
                baseFolder = cmd.getOptionValue("input-folder");
                System.out.println(baseFolder);
            }
            if (cmd.hasOption("o")) {
                outFile = cmd.getOptionValue("output-file");
                System.out.println(outFile);
            }

        } catch (ParseException e) {
            System.out.println(e.getMessage());
            helper.printHelp("Usage:", options);
            System.exit(0);
        }

        System.out.println("Extracting tables");

//        String base_path = "../../../datasets";

//        String baseFolder = "../../datasets/pdf_con_tabelle/";
//        String outFile = "../datasets/test_tabelle.json";

//        Path baseFolder = Paths.get(base_path, "delibere_giunta");
//        Path outFile = Paths.get(base_path, "raw_delibere_giunta.json");

        try {
            readPDF(baseFolder, outFile);
        }
        catch (Exception e){
            System.out.println("error" + e);
        }

        System.out.println("end");
    }

}
