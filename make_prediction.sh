#!/bin/bash

curl \
-H "Content-Type: application/json" \
-X POST \
-d '{"features":[8.3,41,6.9,1.0,322,2.5,37.88,-122.23]}' \
http://127.0.0.1:5000/predict