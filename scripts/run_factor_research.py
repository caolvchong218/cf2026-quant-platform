"""Reproduce the auxiliary factor route without replacing the V3 mainline."""
import argparse
from pathlib import Path
from cfquant.research_lab import run_research,publish_evidence

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-root',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--demo',action='store_true')
    parser.add_argument('--publish',action='store_true')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    output=run_research(root,args.data_root,args.output,demo=args.demo)
    if args.publish:
        if args.demo:parser.error('Synthetic demo cannot replace published real evidence')
        publish_evidence(root,output)
    print(output)
