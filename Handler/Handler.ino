void setup() {
    Serial.begin(9600);
    pinMode(13, OUTPUT);
}

void loop() {
    int received = Serial.read();
    if (received) {
        digitalWrite(13, HIGH);
    } else {
        digitalWrite(13, LOW);
    }
    delay(200);
}