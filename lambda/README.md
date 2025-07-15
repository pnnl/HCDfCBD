# Lambda Container (DEPRECATED)

## Build and run
A containerized version of this code is in the `lambda/` folder.  The container can be built by installing docker and calling:

```bash
docker build -t diomics:latest .
```

Once the image is built, you can test it locally by running and pinging a container.  First, we run the container, mounting the data folder to the container's `/var/task/data` folder, and exposing port 9000 mapping to the container's port 8080.

```bash
docker run -p 9000:8080 -v /Users/clab683/git_repos/DeepIMV/data:/var/task/data diomics:latest
```

The mounted data folder must contain the data files that will be used for training and testing.  The folder structure looks like:
```
data/
    fmeta.csv
    edata/
        edata1.csv
        edata2.csv
        ...
    model.pt
```

With `fmeta.csv` being a valid f_meta file containing sample name alignment for the various edata files in the `edata` subfolder.  `model.pt` must only be included if the `--do-train` flag is not specified and must be a model that is amenable to dimension of the data passed in.

## Test using curl/wget
And tested using a curl/wget command similar to the example from the AWS docs, but passing in arguments using the `--data` flag. 

**httr2**:

```r
library(httr2)

url <- "http://localhost:9000/2015-03-31/functions/function/invocations"

req <- request(url)

headers = c(
  'Content-Type' = 'text/plain'
)

body = list(
    do_train = "true",
    do_eval = "true",
    fmeta_sample_names = c("SampleID_proteomics", "SampleID_lipidpos", "SampleID_metab"),
    fmeta_target_name = "Virus",
    edata_filenames = c("OMICS_ICL104_Metabolomics_YMK.csv", "ICL104_lipids_aligned_for_stats.csv", "ICL104_proteins_luke.csv"),
    edata_cnames = c("Metabolite", "Name", "Protein"),
    shapley = "true",
    igrads = "true"
)

req <- req |>
    req_headers(
        'Content-Type' = 'text/plain'
    )  |>
    req_body_json(body)

result <- req |> req_method("GET") |> req_perform()
```

**curl:**

```bash
curl --location --request GET 'http://localhost:9000/2015-03-31/functions/function/invocations' \
--header 'Content-Type: text/plain' \
--data '{
    "do_train": "true",
    "do_eval": "true",
    "fmeta_sample_names": ["SampleID_proteomics", "SampleID_lipidpos", "SampleID_metab"],
    "fmeta_target_name": "Virus",
    "edata_filenames": ["OMICS_ICL104_Metabolomics_YMK.csv", "ICL104_lipids_aligned_for_stats.csv", "ICL104_proteins_luke.csv"],
    "edata_cnames": ["Metabolite", "Name", "Protein"],
    "shapley": "true",
    "igrads": "true"
}'
```

**wget:**

```bash
wget --no-check-certificate --quiet \
  --method GET \
  --timeout=0 \
  --header 'Content-Type: text/plain' \
  --body-data '{
    "do_train": "true",
    "do_eval": "true",
    "fmeta_sample_names": ["SampleID_proteomics", "SampleID_lipidpos", "SampleID_metab"],
    "fmeta_target_name": "Virus",
    "edata_filenames": ["OMICS_ICL104_Metabolomics_YMK.csv", "ICL104_lipids_aligned_for_stats.csv", "ICL104_proteins_luke.csv"],
    "edata_cnames": ["Metabolite", "Name", "Protein"],
    "shapley": "true",
    "igrads": "true"
}' \
   'http://localhost:9000/2015-03-31/functions/function/invocations'
```