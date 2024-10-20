import sys, time

import qcfractal.interface as ptl
from pathlib import Path


def rmsd_filter(ds_opt, opt_lot, o_file):

    todelete = []

    di = {}

    for i in ds_opt.df.index:
        try:
            di[i] = ds_opt.get_record(
                name=i, specification=opt_lot
            ).get_final_molecule()
        except:
            "ValidationError" or "TypeError"
            continue

    list_keys = list(di.keys())

    count = 0
    for i in range(len(list_keys)):
        for j in range(i + 1, len(list_keys)):
            count = count + 1
            mol1 = di[list_keys[i]]
            mol2 = di[list_keys[j]]
            rmsd = mol1.align(mol2, atoms_map=True)[1]["rmsd"]
            # print(rmsd)
            if rmsd < 0.25 and rmsd != 0.0:
                if not list_keys[j] in todelete:
                    todelete.append(list_keys[j])
    print_out("List of molecules to delete: {}\n".format(todelete), o_file)

    from collections import Counter

    c = Counter(todelete)
    fin = todelete.copy()
    for key, value in c.items():
        if value > 1:
            fin.remove(key)

    for u in fin:
        del di[u]
    final_keys = list(di.keys())
    return di, final_keys


def be_stoich(
    ds_be, database, small_collection, wat_collection, opt_lot, o_file, client
):

    ds_opt = client.get_collection("OptimizationDataset", database)
    ds_m = client.get_collection("OptimizationDataset", small_collection)
    ds_w = client.get_collection("OptimizationDataset", wat_collection)

    di, final_keys = rmsd_filter(ds_opt, opt_lot, o_file)

    for k in final_keys:
        mol = di[k]
        g = mol.geometry.flatten()
        s = mol.symbols

        m = database.split("_")[0]
        op_m = ds_m.get_record(name=m, specification=opt_lot)
        m_2 = op_m.get_final_molecule()

        w = database.split("_")[1] + "_" + database.split("_")[2]

        m_w = ds_w.get_record(name=w, specification=opt_lot)
        m_1 = m_w.get_final_molecule()
        g1 = m_1.geometry.flatten()
        s1 = m_1.symbols

        n1 = len(s1)

        d = ptl.Molecule(
            symbols=s,
            geometry=g,
            fragments=[list(range(0, n1)), list(range(n1, len(s)))],
        )

        j5 = d.get_fragment(0)  # M1
        j4 = d.get_fragment(1)
        j7 = d.get_fragment(0, 1)  # M2
        j6 = d.get_fragment(1, 0)

        be_cal = {
            "default": [
                (d, 1.0),
                (j4, 1.0),
                (j5, 1.0),
                (j7, -1.0),
                (j6, -1.0),
                (m_1, -1.0),
                (m_2, -1.0),
            ],
            "be_nocp": [(d, 1.0), (m_1, -1.0), (m_2, -1.0)],
            "ie": [(d, 1.0), (j7, -1.0), (j6, -1.0)],
            "de": [(m_1, 1.0), (m_2, 1.0), (j4, -1.0), (j5, -1.0)],
        }
        ds_be.add_rxn(k, be_cal)
        ds_be.save()


def compute_be(
    wat_collection,
    small_collection,
    database,
    opt_lot,
    lot,
    o_file,
    be_tag,
    client,
    program='psi4',
):
    name_be = "be_" + str(database) + "_" + opt_lot.split("_")[0]

    try:
        ds_be = ptl.collections.ReactionDataset(
            name_be, ds_type="rxn", client=client, default_program=program
        )
        ds_be.save()

        be_stoich(
            ds_be,
            database,
            small_collection,
            wat_collection,
            opt_lot,
            o_file,
            client=client,
        )

    except KeyError:
        print_out("Be database {} already exists\n".format(str(name_be)), o_file)

        ds_be = client.get_collection("ReactionDataset", name_be)

    mol_id = ds_be.get_entries().loc[0].molecule
    mult = client.query_molecules(mol_id)[0].molecular_multiplicity
    for l in lot:
        print(l)
        if mult == 2:
            keywords = ptl.models.KeywordSet(values={"reference": "uks"})
            try:
                ds_be.add_keywords("rad_be", "psi4", keywords, default=True)
                ds_be.save()
            except KeyError:
               pass

            c = ds_be.compute(
                l.split("_")[0],
                l.split("_")[1],
                keywords="rad_be",
                stoich="default",
                tag=be_tag,
                program=program,
            )
        else:
            c = ds_be.compute(
                l.split("_")[0],
                l.split("_")[1],
                stoich="default",
                tag=be_tag,
                program=program,
            )
    print_out("Collection {}: {}\n".format(name_be, c), o_file)

def compute_hessian(
    be_collection,
    opt_lot,
    o_file,
    hess_tag,
    client,
    program='psi4',
    ):

    try:
        ds_be = client.get_collection("ReactionDataset", be_collection)
    except:
        "KeyError"
        print_out("Reaction  database {} does not exist\n".format(str(be_collection)), o_file)
        return None

    print_out(f"Computing hessian for all molecules in collection {ds_be.name}", o_file)
    df_all = ds_be.get_entries()
    mols = df_all[df_all['stoichiometry'] == 'be_nocp']['molecule'] 
    print_out(f"Molecule list for hessian compute: {mols}", o_file)

    mol_id = ds_be.get_entries().loc[1].molecule
    mult = client.query_molecules(mol_id)[0].molecular_multiplicity

    if float(mult) == 2:
        kw = ptl.models.KeywordSet(**{"values": {'function_kwargs': {'dertype': 1}, 'reference': 'uks'}})
        method = opt_lot.split("_")[0][1:]
    else:
        kw = ptl.models.KeywordSet(**{"values": {'function_kwargs': {'dertype': 1}}})
        method = opt_lot.split("_")[0]

    kw_id = client.add_keywords([kw])[0]
    r = client.add_compute(program, method, opt_lot.split("_")[1], "hessian", kw_id, list(mols), tag=hess_tag)
    print_out(f"{r} hessian computations have been sent at the {method} level of theory.\n", o_file)


