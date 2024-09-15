import argparse
import os
import time
import datetime
from deepsramp import *

parser = argparse.ArgumentParser(description='Run MultiSRAMP')

parser.add_argument('-f', '--file', required=True, type=str, help='Path of datafile')
parser.add_argument('-o', '--out', required=True, type=str, help='Directory of results')
parser.add_argument('-m', '--mode', default='full', type=str, help='Mode of MultiSRAMP')
parser.add_argument('-l', '--len', default=400, type=int, help='Half sequence length of MultiSRAMP')
parser.add_argument('-t', '--target', type=str, help='Target cell line')

args = parser.parse_args()

pid = os.getpid()
print('pid: ', pid)
print(datetime.datetime.now())

# main
i = args.target
if i in ['mature', 'full']:
    traindf, testdf = utils.load(args.file)
else:
    traindf, testdf = utils.load(args.file)[i]

trainds = df2ds_multi(traindf)
traindl = ds2dl(trainds, batch_size=128, drop_last=True, num_workers=4)

testds = df2ds_multi(testdf)
testdl = ds2dl(testds, batch_size=128, shuffle=False, num_workers=4)

device = "cuda" if torch.cuda.is_available() else "cpu"
utils.setup_seed(42)

model = MultiSRAMP(mode=args.mode, halfseqlen=args.len).to(device)
loss_fn = TriLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 1, 2)

print(f'Start to train - [{i}]')
start = time.time()
train(model, loss_fn, optimizer, scheduler, traindl, testdl, epochs=31, device=device)
end = time.time()
test_res = [test_loop(testdl, model, device, loss_fn)]
print(end-start)

torch.save(model.state_dict(), f'{args.out}/{args.mode}_{args.len}_{i}.model')
utils.save(test_res, f'{args.out}/{args.mode}_{args.len}_{i}.metrics')

