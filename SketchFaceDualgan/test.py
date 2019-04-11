import gan
import wgan
import dualgan
import p2p

import os
import sys


# Make directories
if not os.path.exists('gan_results/'):
    os.makedirs('gan_results')
if not os.path.exists('wgan_results/'):
    os.makedirs('wgan_results')
if not os.path.exists('dualgan_results/'):
    os.makedirs('dualgan_results')
if not os.path.exists('p2p_results/'):
    os.makedirs('p2p_results')

# Save standard output
sys_out = sys.stdout

# Test gan
print("testing gan")
output = open("gan_results/terminal.txt", 'w+')
sys.stdout = output
gan.test()
sys.stdout.flush()
sys.stdout = sys_out

# Test wgan
print("testing wgan")
output = open("wgan_results/terminal.txt", 'w+')
sys.stdout = output
wgan.test()
sys.stdout.flush()
sys.stdout = sys_out

# Test dualgan
print("testing dualgan")
output = open("dualgan_results/terminal.txt", 'w+')
sys.stdout = output
dualgan.test()
sys.stdout.flush()
sys.stdout = sys_out

# Test pix2pix
print("testing pix2pix")
output = open("p2p_results/terminal.txt", 'w+')
sys.stdout = output
p2p.test()
sys.stdout.flush()
sys.stdout = sys_out

print("Done.")
