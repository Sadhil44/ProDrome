// Diurnal HTTP load for one nginx service. Part 4.2 of docs/guides/shaurya.md.
//
// One "day" is compressed into 10 minutes: quiet night, morning ramp, midday
// dip, evening peak, back to night. k6 ramps VUs smoothly between stages, so the
// traffic curve is smooth -- that variation is the training signal for the
// detector (guide 4.1: a detector trained on flat load fires on the first real
// traffic change). The 10-minute day repeats DAYS times.
//
//   TARGET  URL to hit             (default http://localhost:8080)
//   DAYS    number of 10-min days  (default 84, ~= 14h)
//
//   k6 run -e TARGET=http://localhost:8080 -e DAYS=84 collect/load/k6-diurnal.js

import http from "k6/http";
import { sleep } from "k6";

const TARGET = __ENV.TARGET || "http://localhost:8080";
const DAYS = Number(__ENV.DAYS || 84);

const DAY = [
  { duration: "2m", target: 5 },   // night
  { duration: "2m", target: 40 },  // morning ramp
  { duration: "2m", target: 25 },  // midday dip
  { duration: "2m", target: 60 },  // evening peak
  { duration: "2m", target: 5 },   // night again
];

export const options = {
  scenarios: {
    diurnal: {
      executor: "ramping-vus",
      startVUs: 5,
      stages: Array.from({ length: DAYS }, () => DAY).flat(),
      gracefulRampDown: "5s",
    },
  },
};

export default function () {
  http.get(TARGET);
  sleep(1);
}
