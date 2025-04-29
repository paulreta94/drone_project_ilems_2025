## Radar detection from drone

This project consists in detecting a high Radar-Cross-Section metal piece, thanks to the OPS243C radar (Omnipresense).
The OPS243C is mounted on a drone and communicates with a PC thanks to MQTT.

In this module, we focus on the distance-waterfall real-time display.

There are different parts :
1) The "node_red_server.json" subscribes to a broker (https://node-red.mattlpt.fr/ui/#!/0?socketid=Ydupu3HsR8TrCoGFAABj
). The publisher is the ESP243 board linked to the OPS243C radar.
2) I and Q frames are retrieved from node_red in a file named "data.json"
3) These frames are read and processed in "test_waterfall.py"
4) The result is displayed in real time in a dash app (see dash.plotly.com)

However, this code is NOT functional : some bugs persist while reading the "data.json" file. The next generation code is then the following.
-> Instead of using a node-red subscriber, one needs to use a mosquitto subscriber (https://github.com/roppert/mosquitto-python-example.git), which runs for example on a RaspberryPi. The goal is then not to use a json file but a First-In-First-Out (FIFO) queue which is retrieved from the subscriber.

For any questions, please send me an e-mail : peretaux@gmail.com