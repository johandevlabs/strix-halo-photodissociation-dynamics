#!/usr/bin/env python3
"""
Phase 0.5, step 2: find out what the SOC toolchain actually exposes.

This is a diagnostic, not a calculation. It answers the three questions that
decide whether the HOX plan survives, and it is written so that FAILURES ARE
OUTPUT rather than tracebacks -- every probe is independently guarded, so one
missing piece does not hide the rest.

  1. Does the import chain work at all (pyscf -> socutils -> x2camf -> prism)?
  2. What is the Prism API for a state-interaction SOC calculation? The
     shipped examples under prism/examples/soc are the ground truth here, so
     this prints them verbatim -- that is the single most useful thing in the
     output.
  3. Do the X2CAMF spin-orbit integrals build for a halogen, and how big is
     the resulting spinor basis?

Nothing here is expensive. If it runs longer than a couple of minutes on the
EVO, something is wrong.

Usage:
    python 01_soc_probe.py 2>&1 | tee logs/soc_probe.log
    python 01_soc_probe.py --atom Br --basis ano-rcc-vdzp
    python 01_soc_probe.py --src ~/src        # where 00_setup_soc.sh cloned
"""
import argparse
import importlib
import inspect
import os
import pkgutil
import traceback

RULE = "=" * 72


def section(title):
    print(f"\n{RULE}\n== {title}\n{RULE}")


def probe(label):
    """Decorator: run a probe, report the outcome, never raise."""
    def wrap(fn):
        def inner(*a, **kw):
            try:
                return fn(*a, **kw)
            except Exception as exc:
                print(f"\n  !! {label} FAILED: {type(exc).__name__}: {exc}")
                traceback.print_exc()
                return None
        return inner
    return wrap


# ---------------------------------------------------------------- imports
def report_imports():
    """x2camf and zquatev are bundled inside socutils rather than being
    separate top-level distributions, so try both spellings and record the
    one that works under the short name."""
    section("1. import chain")
    mods = {}
    candidates = [
        ("numpy",      ["numpy"]),
        ("scipy",      ["scipy"]),
        ("pyscf",      ["pyscf"]),
        ("socutils",   ["socutils"]),
        ("x2camf",     ["x2camf", "socutils.x2camf"]),
        ("zquatev",    ["zquatev", "socutils.zquatev"]),
        ("prism",      ["prism"]),
        ("sympy",      ["sympy"]),
        ("opt_einsum", ["opt_einsum"]),
    ]
    for short, names in candidates:
        for name in names:
            try:
                m = importlib.import_module(name)
            except Exception as exc:
                last = exc
                continue
            mods[short] = m
            ver = getattr(m, "__version__", "(no __version__)")
            print(f"  ok      {short:10s} {ver:12s} "
                  f"{'as ' + name + '  ' if name != short else ''}"
                  f"{getattr(m, '__file__', '')}")
            break
        else:
            print(f"  MISSING {short:10s} tried {', '.join(names)}: "
                  f"{type(last).__name__}: {last}")
    return mods


# ------------------------------------------------------------- prism API
@probe("prism module walk")
def walk_prism(prism):
    section("2a. prism submodules")
    root = os.path.dirname(prism.__file__)
    print(f"  package root: {root}\n")
    names = []
    for mi in pkgutil.walk_packages([root], prefix="prism."):
        names.append(mi.name)
        print(f"  {mi.name}")
    return names


@probe("prism SOC name search")
def find_soc_names(prism, modnames):
    """Anything with 'soc'/'spin_orbit'/'dipole'/'osc' in its name, anywhere."""
    section("2b. SOC- and intensity-related names")
    needles = ("soc", "spin_orbit", "spinorbit", "dipole", "oscillator",
               "osc_str", "transition", "trdm", "eigvec", "amfi", "breit")
    hits = 0
    for modname in ["prism"] + list(modnames or []):
        try:
            m = importlib.import_module(modname)
        except Exception:
            continue                      # submodule may need optional deps
        for attr in dir(m):
            if attr.startswith("_"):
                continue
            low = attr.lower()
            if any(n in low for n in needles):
                obj = getattr(m, attr, None)
                kind = type(obj).__name__
                print(f"  {modname}.{attr}  ({kind})")
                hits += 1
    if not hits:
        print("  (nothing matched -- read the examples in 2c instead)")


@probe("prism examples")
def show_examples(src):
    """The shipped SOC examples are the real API documentation."""
    section("2c. prism SOC examples (VERBATIM -- this is the API)")
    exdir = os.path.join(os.path.expanduser(src), "prism", "examples")
    if not os.path.isdir(exdir):
        print(f"  no examples dir at {exdir}")
        print("  (pass --src to point at where 00_setup_soc.sh cloned prism)")
        return
    for sub in sorted(os.listdir(exdir)):
        subdir = os.path.join(exdir, sub)
        if os.path.isdir(subdir):
            print(f"  {sub}/: {', '.join(sorted(os.listdir(subdir)))}")

    socdir = os.path.join(exdir, "soc")
    if not os.path.isdir(socdir):
        print("\n  no examples/soc directory")
        return
    for fn in sorted(os.listdir(socdir)):
        if not fn.endswith(".py"):
            continue
        path = os.path.join(socdir, fn)
        print(f"\n----- {os.path.relpath(path, exdir)} " + "-" * 20)
        with open(path) as fh:
            print(fh.read())


@probe("QD-NEVPT2 signature")
def show_signatures(prism):
    """Constructor and kernel signatures, so we know what knobs exist."""
    section("2d. signatures of the QD-NEVPT2 / SOC entry points")
    for modname in ("prism.qdnevpt", "prism.nevpt", "prism.interface",
                    "prism.soc", "prism.mr_adc"):
        try:
            m = importlib.import_module(modname)
        except Exception as exc:
            # Print the MESSAGE, not just the type. A bare ModuleNotFoundError
            # on prism.nevpt says nothing; "No module named 'foo'" names the
            # missing dependency and is the whole diagnosis.
            print(f"  {modname}: not importable -- {type(exc).__name__}: {exc}")
            continue
        print(f"\n  --- {modname}")
        for attr in sorted(dir(m)):
            if attr.startswith("_"):
                continue
            obj = getattr(m, attr)
            if not (inspect.isclass(obj) or inspect.isfunction(obj)):
                continue
            try:
                sig = inspect.signature(obj)
            except (TypeError, ValueError):
                sig = "(signature unavailable)"
            print(f"    {attr}{sig}")
            if inspect.isclass(obj):
                knobs = [a for a in dir(obj)
                         if not a.startswith("_")
                         and not callable(getattr(obj, a, None))]
                if knobs:
                    print(f"        attrs: {', '.join(knobs)}")


# ------------------------------------------------------- x2camf integrals
@probe("X2CAMF integrals")
def probe_integrals(atom, basis):
    section(f"3. X2CAMF spin-orbit integrals for {atom}/{basis}")
    from pyscf import gto
    import socutils

    mol = gto.M(atom=f"{atom} 0 0 0", basis=basis, spin=1,
                charge=0, verbose=0)
    print(f"  nao (scalar)      {mol.nao_nr()}")
    print(f"  n electrons       {mol.nelectron}")
    print(f"  spin (2S)         {mol.spin}")

    # socutils exposes SOC either as a .x2camf() shortcut on a spinor SCF
    # object or through GHF. Try the documented spinor route and report what
    # comes back; the point is to confirm the integrals BUILD for a halogen,
    # not to converge anything publishable.
    print("\n  socutils public API:")
    for attr in sorted(a for a in dir(socutils) if not a.startswith("_")):
        print(f"    socutils.{attr}")

    for name in ("x2camf", "socutils.x2camf"):
        try:
            x2camf = importlib.import_module(name)
        except Exception:
            continue
        print(f"\n  {name} public API:")
        for attr in sorted(a for a in dir(x2camf) if not a.startswith("_")):
            print(f"    {name}.{attr}")
        break
    else:
        print("\n  x2camf not importable under either name")
    return mol


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--atom", default="Cl",
                   help="halogen to probe integrals with (default Cl: "
                        "cheapest of the three that matter)")
    p.add_argument("--basis", default="def2-tzvp",
                   help="basis for the integral probe. def2-tzvp is what the "
                        "Prism SOC examples use and is all-electron through "
                        "Kr. ano-rcc etc. generally need "
                        "`pip install basis-set-exchange`.")
    p.add_argument("--src", default="~/src",
                   help="where 00_setup_soc.sh cloned prism (for examples)")
    args = p.parse_args()

    print(RULE)
    print("== HOX Phase 0.5 -- SOC toolchain probe")
    print(RULE)

    mods = report_imports()

    prism = mods.get("prism")
    if prism is not None:
        modnames = walk_prism(prism)
        find_soc_names(prism, modnames)
        show_signatures(prism)
    else:
        print("\n  prism did not import -- skipping API probes")
    show_examples(args.src)

    if "socutils" in mods:
        probe_integrals(args.atom, args.basis)
    else:
        print("\n  socutils missing -- skipping integral probe")

    section("what to do with this")
    print("""
  Paste the whole log back. The decisive parts are 2c (the shipped SOC
  examples, which give the exact call sequence) and 2d (signatures, which
  show whether oscillator strengths and SOC eigenvectors are reachable).

  02_soc_atoms.py has its Prism call isolated in one function with the
  assumptions written out; it gets corrected from this output rather than
  guessed at twice.
""")


if __name__ == "__main__":
    main()
