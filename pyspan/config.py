import ConfigParser
import os
import matplotlib as mpl

# A hack to figure out if I'm on ganesha or not
ganesha = os.environ["USER"] == "sabina"
parent_dir = "Box/" if not ganesha else ""
CONFIG_FILE = os.path.expanduser("~/{}LoP/pyspan/settings".format(parent_dir))

configParser = ConfigParser.RawConfigParser()
configParser.read(CONFIG_FILE)
settings = dict(configParser.items("settings"))
settings["years"] = map(int, settings["years"].split(", "))
settings["percentile"] = float(settings["percentile"])
settings["election_cycles"] = map(int, settings["election_cycles"].split(", "))
settings["debate_types"] = settings["debate_types"].split(", ")
assert settings["mode"] in ("debates", "crec")

mpl.use(settings["mpl_backend"])

paths = configParser.items("PATHS")
if settings["mode"] == "debates":
    paths += configParser.items("DEBATE_PATHS")
elif settings["mode"] == "crec":
    paths += configParser.items("CREC_PATHS")
paths = dict(map(lambda kv: (kv[0], os.path.expanduser(kv[1])+"/"), paths))
pkgdir = paths["package_dir"]
paths["input_dir"] = pkgdir + "inputs/"
paths["output_dir"] = pkgdir + "output/"

debate_paths = dict(map(lambda kv: (kv[0], os.path.expanduser(kv[1])+"/"),
                        configParser.items("DEBATE_PATHS")))
crec_paths = dict(map(lambda kv: (kv[0], os.path.expanduser(kv[1])+"/"),
                        configParser.items("CREC_PATHS")))

headers = dict(configParser.items("headers"))
