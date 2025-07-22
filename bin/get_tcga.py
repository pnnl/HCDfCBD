"""
Scrape the TCGA dataset from the Broad Institute's website.  Make sure to change 'out_dir' below to wherever you want the data to be stored.
"""

import subprocess
import os

tumor_list = [
'ACC',
'BLCA',
'BRCA',
'CESC',
'CHOL',
'COAD',
'COADREAD',
'DLBC',
'ESCA',
'FPPP',
'GBM',
'GBMLGG',
'HNSC',
'KICH',
'KIPAN',
'KIRC',
'KIRP',
'LAML',
'LGG',
'LIHC',
'LUAD',
'LUSC',
'MESO',
'OV',
'PAAD',
'PCPG',
'PRAD',
'READ',
'SARC',
'SKCM',
'STAD',
'STES',
'TGCT',
'THCA',
'THYM',
'UCEC',
'UCS',
'UVM']

orgi_wd = os.getcwd()
out_dir = "~/Data/tcga/"
os.chdir(out_dir)

try:
    output = subprocess.check_output(["wget", "-O", out_file, url], stderr=subprocess.STDOUT)
except subprocess.CalledProcessError as exc:
    print(exc.output)
else:
    assert 0

try:
    output = subprocess.check_output(["which", "wget"], stderr=subprocess.STDOUT)
except subprocess.CalledProcessError as exc:
    print(exc.output)
else:
    assert 0

# RPPA
rppa_out_dir = os.path.join(out_dir, "RPPA")
os.makedirs(rppa_out_dir, exist_ok=True)

for tumor in tumor_list:
    url = f"https://gdac.broadinstitute.org/runs/stddata__2016_01_28/data/{tumor}/20160128/gdac.broadinstitute.org_{tumor}.RPPA_AnnotateWithGene.Level_3.2016012800.0.0.tar.gz"
    out_file = os.path.join(rppa_out_dir, os.path.basename(url))
    result = subprocess.call(["wget", "-O", out_file, url])
    if not result == 0:
        print("No RPPA data for {}".format(tumor))
        os.remove(out_file)
    else:
        fname = f"gdac.broadinstitute.org_{tumor}.RPPA_AnnotateWithGene.Level_3.2016012800.0.0.tar.gz"
        result = subprocess.call(["tar", "-xzvf", out_file, "-C", rppa_out_dir])

        if not result == 0:
            raise Exception("Error extracting file")

# miRNA Seq
mirna_out_dir = os.path.join(out_dir, "miRNA")
os.makedirs(mirna_out_dir, exist_ok=True)

for tumor in tumor_list:
    url = f"https://gdac.broadinstitute.org/runs/stddata__2016_01_28/data/{tumor}/20160128/gdac.broadinstitute.org_{tumor}.miRseq_Preprocess.Level_3.2016012800.0.0.tar.gz"
    out_file = os.path.join(mirna_out_dir, os.path.basename(url))
    result = subprocess.call(["wget", "-O", out_file, url])
    if not result == 0:
        print("No miRNA data for {}".format(tumor))
        os.remove(out_file)
    else:
        fname = f"gdac.broadinstitute.org_{tumor}.miRseq_Preprocess.Level_3.2016012800.0.0.tar.gz"
        result = subprocess.call(["tar", "-xzvf", out_file, "-C", mirna_out_dir])

        if not result == 0:
            raise Exception("Error extracting file")
        else:
            os.remove(out_file)

# methylation
methyl_out_dir = os.path.join(out_dir, "methylation")
os.makedirs(methyl_out_dir, exist_ok=True)

for tumor in tumor_list:
    url = f"https://gdac.broadinstitute.org/runs/stddata__2016_01_28/data/{tumor}/20160128/gdac.broadinstitute.org_{tumor}.Methylation_Preprocess.Level_3.2016012800.0.0.tar.gz"
    out_file = os.path.join(methyl_out_dir, os.path.basename(url))
    result = subprocess.call(["wget", "-O", out_file, url])
    if not result == 0:
        print("No methylation data for {}".format(tumor))
        os.remove(out_file)
    else:
        fname = f"gdac.broadinstitute.org_{tumor}.miRseq_Preprocess.Level_3.2016012800.0.0.tar.gz"
        result = subprocess.call(["tar", "-xzvf", out_file, "-C", methyl_out_dir])

        if not result == 0:
            raise Exception("Error extracting file")
        else:
            os.remove(out_file)

# rna-seq
rnaseq_out_dir = os.path.join(out_dir, "rnaseq")
os.makedirs(rnaseq_out_dir, exist_ok=True)

for tumor in tumor_list[14:]:
    print("pulling data for {}".format(tumor))
    url = f"https://gdac.broadinstitute.org/runs/stddata__2016_01_28/data/{tumor}/20160128/gdac.broadinstitute.org_{tumor}.mRNAseq_Preprocess.Level_3.2016012800.0.0.tar.gz"
    out_file = os.path.join(rnaseq_out_dir, os.path.basename(url))
    result = subprocess.call(["wget", "-O", out_file, url])
    if not result == 0:
        print("No rnaseq data for {}".format(tumor))
        os.remove(out_file)
    else:
        fname = f"gdac.broadinstitute.org_{tumor}.mRNAseq_Preprocess.Level_3.2016012800.0.0.tar.gz"
        result = subprocess.call(["tar", "-xzvf", out_file, "-C", rnaseq_out_dir])

        if not result == 0:
            raise Exception("Error extracting file")
        else:
            os.remove(out_file)

# clinical labels
labels_dir = os.path.join(out_dir, "clinical_labels")
os.makedirs(labels_dir, exist_ok=True)

for tumor in tumor_list:
    print("pulling data for {}".format(tumor))
    url = f"https://gdac.broadinstitute.org/runs/stddata__2016_01_28/data/{tumor}/20160128/gdac.broadinstitute.org_{tumor}.Clinical_Pick_Tier1.Level_4.2016012800.0.0.tar.gz"
    out_file = os.path.join(labels_dir, os.path.basename(url))
    result = subprocess.call(["wget", "-O", out_file, url])
    if not result == 0:
        print("No clinical data for {}".format(tumor))
        os.remove(out_file)
    else:
        fname = f"gdac.broadinstitute.org_{tumor}.Clinical_Pick_Tier1.Level_4.2016012800.0.0.tar.gz"
        result = subprocess.call(["tar", "-xzvf", out_file, "-C", labels_dir])

        if not result == 0:
            raise Exception("Error extracting file")
        else:
            os.remove(out_file)


