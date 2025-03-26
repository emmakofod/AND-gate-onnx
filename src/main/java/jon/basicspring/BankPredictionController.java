package jon.basicspring;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import java.io.InputStream;
import java.util.Collections;

@Controller
public class BankPredictionController {

    @GetMapping("/bank")
    public String showForm() {
        return "bank-form";
    }

    @PostMapping("/bank")
    public String submitForm(
            @RequestParam("number1") double creditScore,
            @RequestParam("number2") double age,
            @RequestParam("number3") double tenure,
            @RequestParam("number4") double balance,
            @RequestParam("number5") double numOfProducts,
            @RequestParam("number6") double hasCrCard,
            @RequestParam("number7") double isActiveMember,
            @RequestParam("number8") double estimatedSalary,
            @RequestParam("number9") double geographyFrance,
            @RequestParam("number10") double geographyGermany,
            @RequestParam("number11") double geographySpain,
            @RequestParam("number12") double genderFemale,
            @RequestParam("number13") double genderMale,
            Model model) {

        // Store all input values in the model
        model.addAttribute("creditScore", creditScore);
        model.addAttribute("age", age);
        model.addAttribute("tenure", tenure);
        model.addAttribute("balance", balance);
        model.addAttribute("numOfProducts", numOfProducts);
        model.addAttribute("hasCrCard", hasCrCard);
        model.addAttribute("isActiveMember", isActiveMember);
        model.addAttribute("estimatedSalary", estimatedSalary);
        model.addAttribute("geographyFrance", geographyFrance);
        model.addAttribute("geographyGermany", geographyGermany);
        model.addAttribute("geographySpain", geographySpain);
        model.addAttribute("genderFemale", genderFemale);
        model.addAttribute("genderMale", genderMale);

        // Prepare input for ONNX model
        double[] inputFeatures = {creditScore, age, tenure, balance, numOfProducts, hasCrCard, isActiveMember, estimatedSalary,
                geographyFrance, geographyGermany, geographySpain, genderFemale, genderMale};

        // Get the raw model output and calculate final prediction
        double predictionRaw = runInference(inputFeatures);
        model.addAttribute("predictionRaw", predictionRaw);

        // If the raw prediction is above 0.5, predict '1' (likely to leave); otherwise, '0' (likely to stay)
        int prediction = predictionRaw > 0.5 ? 1 : 0;
        model.addAttribute("prediction", prediction);

        return "bankresult";
    }

    private double runInference(double[] inputFeatures) {
        String modelPath = "/bank_model.onnx";
        try (InputStream is = BankPredictionController.class.getResourceAsStream(modelPath);
             OrtEnvironment env = OrtEnvironment.getEnvironment()) {
            assert is != null;
            OrtSession session = env.createSession(is.readAllBytes());

            float[][] inputData = new float[][]{convertToFloatArray(inputFeatures)};
            OnnxTensor inputTensor = OnnxTensor.createTensor(env, inputData);
            OrtSession.Result result = session.run(Collections.singletonMap("input", inputTensor));

            float[][] output = (float[][]) result.get(0).getValue();
            return output[0][0]; // Return the raw prediction value
        } catch (Exception e) {
            throw new RuntimeException("Error running ONNX inference", e);
        }
    }

    private float[] convertToFloatArray(double[] inputFeatures) {
        float[] floatArray = new float[inputFeatures.length];
        for (int i = 0; i < inputFeatures.length; i++) {
            floatArray[i] = (float) inputFeatures[i];
        }
        return floatArray;
    }
}
