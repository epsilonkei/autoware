#!/usr/bin/env bash

if [ ! "$XRANGE" ]; then XRANGE="[*:*]"; fi
if [ ! "$YRANGE" ]; then YRANGE="[*:*]"; fi

echo "XRANGE=$XRANGE";
echo "YRANGE=$YRANGE";

gnuplot <<EOF
# set terminal postscript eps color enhanced
set terminal png size 1280, 900
set tics font "Times New Roman,16"
set xlabel font "Times New Roman,16"
set ylabel font "Times New Roman,16"
set zlabel font "Times New Roman,16"
set key font "Times New Roman,16"
set key right top
set key width 3
# set output "train_log.eps"
set output "train_log.png"
set grid
set size ratio 0.5
set xlabel "Epoch"
set ylabel "Loss"
set xrange ${XRANGE}
set title "_"
set logscale y
onlySim_vel_loss(x) = 2.424530e-02
onlySim_steer_loss(x) = 2.485642e-05
onlySim_dsteer_loss(x) = 1.847670e-03
plot "train_log.txt" using 1:2 with line linewidth 2 title "VelLoss", "train_log.txt" using 1:3 with line linewidth 2 title "SteerLoss", onlySim_vel_loss(x) with lines linewidth 2 title "OnlySimulateVelLoss", onlySim_steer_loss(x) with lines linewidth 2 title "OnlySimulateSteerLoss", "train_log.txt" using 1:4 with line linewidth 2 title "dSteerLoss", onlySim_dsteer_loss(x) with lines linewidth 2 title "OnlySimulateDsteerLoss"
EOF
