# Radar detection from drone
Robert Drouineau, Nicolas Soto and Paul-Etienne Rétaux - teacher : Mr. Yann 
April 2025, ENSTA Paris

## Introduction
This project consists in detecting a high Radar-Cross-Section metal piece, thanks to the OPS243C radar (Omnipresense).
The OPS243C is mounted on a drone and communicates with a PC thanks to MQTT.

There are three main modules detailed below. Each of the module uses plotly (https://plotly.com/python) and opens a window in your favorite browser.

## real_time_waterfall.py 

There are different parts :
1) The "node_red_server.json" subscribes to a broker (https://node-red.mattlpt.fr/ui/#!/0?socketid=Ydupu3HsR8TrCoGFAABj
). The publisher is the ESP243 board linked to the OPS243C radar.
2) I and Q frames are retrieved from node_red in a file named "data.json"
3) These frames are read and processed in "test_waterfall.py"
4) The result is displayed in real time in a dash app (see dash.plotly.com)

However, this code is NOT functional : some bugs persist while reading the "data.json" file. The next generation code is then the following.
-> Instead of using a node-red subscriber, one needs to use a mosquitto subscriber (https://github.com/roppert/mosquitto-python-example.git), which runs for example on a RaspberryPi. The goal is then not to use a json file but a First-In-First-Out (FIFO) queue which is retrieved from the subscriber.

## s_matrix_samples.py

This module plots the absolute value of the s vector (np.conj(I + jQ)). There are as many plots as couples of I,Q vectors. 

## test_waterfall_map.py

This module plots the distance waterfall obtained from the s vector. You can test the file with two JSON raw-data-files : Test1_5m.json or Test1_10m.json.

For any questions, please send me an e-mail : peretaux@gmail.com