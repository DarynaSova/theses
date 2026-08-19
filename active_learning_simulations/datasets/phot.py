def edit_disorder_fasta(input_file, output_file):

    with open(input_file) as infile, open(output_file, "w") as outfile:

        for line in infile:
            line = line.strip()

            if line.startswith(">"):
                # Remove ">" and split header into fields
                header = line[1:].strip()
                parts = header.split()

                # First field is the sequence/mutation name
                name = "mut_" + parts[0]

                # Keep everything except SET=...
                remaining = [
                    part for part in parts[1:]
                    if not part.startswith("SET=")
                ]

                # Reconstruct header
                new_header = ">" + name + " " + " ".join(remaining)

                outfile.write(new_header + "\n")

            elif line:
                # Disorder scores: leave unchanged
                outfile.write(line + "\n")


# Change only the filenames here
edit_disorder_fasta(
    input_file="PHOT_CHLRE_Chen_2023_max2000.fasta.disorder",
    output_file="PHOT_CHLRE_Chen_2023_max2000.fasta.disorder.fixed"
)