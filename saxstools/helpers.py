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
            if element == 'H' and hydrogen:
                continue

            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])

            elements.append(element)
            coordinates.append([x, y, z])

    return elements, coordinates
