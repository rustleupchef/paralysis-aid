import java.io.BufferedReader;
import java.io.FileReader;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.HashMap;
import java.awt.Robot;
import java.awt.event.KeyEvent;


import com.google.gson.Gson;

class jsonObject {
    public String dir;
    public String[] squint;
    public String[] smirk;
    public boolean open_mouth;
    public String eeg_class;
    public float confidence;

    jsonObject() {

    }

    jsonObject (String dir, String[] squint, String[] smirk, boolean open_mouth, String eeg_class, float confidence) {
        this.dir = dir;
        this.squint = squint;
        this.smirk = smirk;
        this.open_mouth = open_mouth;
        this.eeg_class = eeg_class;
        this.confidence = confidence;
    }
}

class typingObject {
    public String[] vertical;
    public String[] horizontal;
    public String[] blink;
    public String[] smirk;
    public String mouth;
    public HashMap<String, String> eeg;

    typingObject() {

    }

    typingObject (String[] vertical, String[] horizontal, String[] blink, String[] smirk, String mouth, HashMap<String, String> eeg) {
        this.vertical = vertical;
        this.horizontal = horizontal;
        this.blink = blink;
        this.smirk = smirk;
        this.mouth = mouth;
        this.eeg = eeg;
    }
}

class transferObject {
    public String vertical;
    public String horizontal;
    public String leftBlink;
    public String rightBlink;
    public String leftSmirk;
    public String rightSmirk;
    public String mouth;
    public String eeg;

    transferObject() {

    }

    transferObject (jsonObject reading, typingObject Typing) {
        String[] dirParts = reading.dir.split("-");
        int horiz = dirParts[0].toLowerCase().equals("left") ? 0 : dirParts[0].toLowerCase().equals("right") ? 1 : 2;
        int vert = dirParts[1].toLowerCase().equals("up") ? 0 : dirParts[1].toLowerCase().equals("down") ? 1 : 2;

        this.vertical = vert < 2 ? Typing.vertical[vert] : "";
        this.horizontal = horiz < 2 ? Typing.horizontal[horiz] : "";

        this.leftBlink = reading.squint[0].equals("True") ? Typing.blink[0] : "";
        this.rightBlink = reading.squint[1].equals("True") ? Typing.blink[1] : "";

        this.leftSmirk = reading.smirk[0].equals("True") ? Typing.smirk[0] : "";
        this.rightSmirk = reading.smirk[1].equals("True") ? Typing.smirk[1] : "";

        this.mouth = reading.open_mouth ? Typing.mouth : "";

        this.eeg = Typing.eeg.getOrDefault(reading.eeg_class, "");
    }

    transferObject (String vertical, String horizontal, String leftBlink, String rightBlink, String leftSmirk, String rightSmirk, String mouth, String eeg) {
        this.vertical = vertical;
        this.horizontal = horizontal;
        this.leftBlink = leftBlink;
        this.rightBlink = rightBlink;
        this.leftSmirk = leftSmirk;
        this.rightSmirk = rightSmirk;
        this.mouth = mouth;
        this.eeg = eeg;
    }

    public String condense() {
        return this.vertical + this.horizontal + this.leftBlink + this.rightBlink + this.leftSmirk + this.rightSmirk + this.mouth + this.eeg;
    }
}

public class App {
    public static void main(String[] args) throws Exception {

        Robot robot = new Robot();
        robot.setAutoDelay(40);
        typingObject Typing = new Gson().fromJson(new FileReader("typing.json"), typingObject.class);

        while (true) {
            URL url = new URL("http://localhost:3000/mindwave/detection");
            HttpURLConnection connection = (HttpURLConnection) url.openConnection();
            connection.setRequestMethod("GET");

            jsonObject reading = new jsonObject();
            int code = connection.getResponseCode();
            if (code == HttpURLConnection.HTTP_OK) {
                BufferedReader in = new BufferedReader(new InputStreamReader(connection.getInputStream()));
                String text = "";
                String line;
                while ((line = in.readLine()) != null) {
                    text += line + "\n";
                }
                in.close();
                text = text.substring(0, text.length() - 1);
                reading = new Gson().fromJson(text, jsonObject.class);
            } else {
                continue;
            }

            transferObject transfer = new transferObject(reading, Typing);
            System.out.println(transfer.condense());
            waylandTypeString(transfer.condense());
        }
    }

    public static void waylandTypeString(String text) throws Exception {
        Process p = new ProcessBuilder("ydotool", "type", text).start();
        p.waitFor();
        Thread.sleep(100);
    }

    public static void typeString(Robot robot, String text) throws InterruptedException {
        System.out.println(text);
        for (char character : text.toCharArray()) {
            System.out.println(character);
            int keyCode = KeyEvent.getExtendedKeyCodeForChar(character);
            
            if (Character.isUpperCase(character) || needsShift(character)) {
                robot.keyPress(KeyEvent.VK_SHIFT);
                robot.keyPress(keyCode);
                robot.keyRelease(keyCode);
                robot.keyRelease(KeyEvent.VK_SHIFT);
            } else {
                robot.keyPress(keyCode);
                robot.keyRelease(keyCode);
            }
            robot.delay(40); 
        }
    }

    private static boolean needsShift(char c) {
        return c == '!' || c == '@' || c == '#' || c == '$' || c == '%' || c == '^' || 
               c == '&' || c == '*' || c == '(' || c == ')' || c == ':' || c == '"' || 
               c == '?' || c == '<' || c == '>';
    }
}
