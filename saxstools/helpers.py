from numpy import loadtxt, asarray, float64, unique, ones


def parse_pdb(pdbfile, hydrogen=False):
    """Simple PDB-file parser to extract elements and coordinates."""

    elements = []
    coordinates = []
    with open(pdbfile) as pdb:
        for line in pdb:
            # only ATOM lines are of interest to us
            if not line.startswith('ATOM'):
                continue

            element = line[76:78].strip()
            # if element is empty, use the first letter of the
            # atomname as element
            if not element:
                element = line[12:16].strip()[0]

            # no hydrogens
            if element == 'H' and not hydrogen:
                continue

            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])

            elements.append(element)
            coordinates.append([x, y, z])

    return elements, asarray(coordinates, dtype=float64)


def parse_saxsdata(infile):

    data = loadtxt(infile, comments='#')

    if data.shape[-1] not in (2, 3):
        raise IOError('File has no proper SAXS data.')
    elif data.shape[-1] == 2:
        q, Iq = data.T
        sq = 0.01 * Iq
    else:
        q, Iq, sq = data.T

    return q, Iq, sq


def coarse_grain(structure, bpr=2):

    beads = []
    center_of_mass = []
    for c in structure.chain_list:
        chain = structure.select('chain', c)
        for resi in unique(chain.data['resi']):
            residue = chain.select('resi', resi)
            resn = residue.sequence[0]
            com = residue.center_of_mass

            # ALA and GLY only have one bead
            if bpr == 2 and resn not in ('ALA', 'GLY'):
                center_of_mass.append(com)
                beads.append('BB')

            center_of_mass.append(com)
            beads.append(resn)

    return beads, asarray(center_of_mass, dtype=float64)
