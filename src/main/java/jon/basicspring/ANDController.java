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
import java.util.Arrays;
import java.util.Collections;

@Controller
public class ANDController {

    @GetMapping("/numbers")
    public String showForm(Model model) {
        return "numbers-form";
    }

    @PostMapping("/numbers")
    public String submitForm(@RequestParam("number1") double number1, @RequestParam("number2") double number2, Model model) {
        model.addAttribute("number1", number1);
        model.addAttribute("number2", number2);
        model.addAttribute("prediction", runInference(number1,number2));
        return "result";
    }

    String runInference(double a, double b){
        String modelPath = "/xorPrev.onnx";
        try (InputStream is = ANDController.class.getResourceAsStream(modelPath);
             OrtEnvironment env = OrtEnvironment.getEnvironment()) {
            assert is != null;
            OrtSession session = env.createSession(is.readAllBytes());
            // scaler her...
            float[][] inputData = new float[][]{{(float) a, (float) b}};  // Your input data
            OnnxTensor inputTensor = OnnxTensor.createTensor(env, inputData);
            OrtSession.Result result = session.run(Collections.singletonMap("dense_input", inputTensor));

            float[][] output = (float[][]) result.get(0).getValue();
            return "" + Arrays.toString(output[0]);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

}
