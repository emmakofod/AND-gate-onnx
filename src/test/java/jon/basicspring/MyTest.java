package jon.basicspring;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.SpringBootTest;

@SpringBootTest
public class MyTest {


    @Test
    public void myTest(){
        Assertions.assertNull(null);
    }
}
