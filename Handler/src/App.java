import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;

import com.fazecast.jSerialComm.SerialPort;
import com.google.gson.Gson;

class jsonObject {
    public String dir;
    public String[] squint;
    public String[] smirk;
    public boolean open_mouth;
    public String eeg_class;

    jsonObject() {

    }

    jsonObject (String dir, String[] squint, String[] smirk, boolean open_mouth, String eeg_class) {
        this.dir = dir;
        this.squint = squint;
        this.smirk = smirk;
        this.open_mouth = open_mouth;
        this.eeg_class = eeg_class;
    }
}

public class App {
    public static void main(String[] args) throws Exception {
        SerialPort port = SerialPort.getCommPort("/dev/ttyACM0");
        if (!port.openPort()) {
            System.out.println("Port didn't open succesfully");
            return;
        }
        port.setBaudRate(9600);
        Thread.sleep(2000);

        OutputStream out = port.getOutputStream();

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

            if (!port.isOpen()) {
                System.out.println("Port disconnected, attempting to reconnect...");
                if (!port.openPort()) {
                    System.out.println("Failed to reconnect");
                    continue;
                }
                out = port.getOutputStream();
            }
            
            try {
                System.out.println(reading.smirk[0]);
                out.write(reading.smirk[0].equals("True") ? 1 : 0);
                out.flush();
            } catch (Exception e) {
                System.out.println("Write failed: " + e.getMessage());
                if (port.isOpen()) {
                    port.closePort();
                }
            }
            Thread.sleep(1000);
        }
    }
}
