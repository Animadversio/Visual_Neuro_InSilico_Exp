from pptx.util import Inches, Length, Pt
from os.path import join
from tqdm import tqdm

protodir = r"F:\insilico_exps\CorNet-recurrent-evol\proto_summary"
tracedir = r"F:\insilico_exps\CorNet-recurrent-evol\actdyn_summary"


def merge_plots2slides(area, sublayer, chanrng, outdir):
    for channum in tqdm(range(chanrng[0], chanrng[1])):
        chanlabel = f"{area}-{sublayer}-Ch{channum:03d}"
        tracepath = (join(tracedir, f"{chanlabel}_act_traces.png"))
        hmappath = (join(tracedir, f"{chanlabel}_act_heatmap.png"))
        protopath = (join(protodir, f"{chanlabel}_allproto_mtg.jpg"))

