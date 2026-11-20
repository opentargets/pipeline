
            #!/bin/bash
            set -euo pipefail
            set -v

mkfs.ext4 -F /dev/disk/by-id/google-work-disk
mkdir -p /mnt/disks/work
mount -o discard,defaults /dev/disk/by-id/google-work-disk /mnt/disks/work
chmod a+w /mnt/disks/work

            useradd -m app
            usermod -a -G docker app
            mkdir -p /home/app
            cd /home/app
            orig=( "gs://a-bucket/a/config.yaml" )
            dest=( "/config.yaml" )
            if [ -n "${dest[*]}" ]; then
                for d in "${!dest[@]}"; do
                    if=${orig[$d]}
                    od=$(dirname "${dest[$d]}")
                    mkdir -p "$od"
                    of=${dest[$d]}
                    echo "Copying $if to $of ($od)"
                    sudo -u app docker run                         --rm                         -v /home/app/"$od":/downloads/"$od"                         gcr.io/google.com/cloudsdktool/google-cloud-cli:588.0.0-alpine@sha256:44924be6dcf684d4e76922c5cb015950305e8ded844587f915dbaeefe290f545                         gcloud storage cp "$if" /downloads/"$of"
                done
            fi
            secret_names=( "a-secret" )
            secret_dest=( "/secrets/a-secret" )
            if [ -n "${secret_dest[*]}" ]; then
                for s in "${!secret_dest[@]}"; do
                    sn=${secret_names[$s]}
                    sd=${secret_dest[$s]}
                    sf="/home/app/${sd#/}"
                    stmp="${sf}.tmp"
                    sod=$(dirname "$sf")
                    mkdir -p "$sod"
                    if command -v gcloud >/dev/null 2>&1; then
                        gcloud secrets versions access latest --secret="$sn" > "$stmp"
                    else
                        sudo -u app docker run                             --rm                             --network host                             gcr.io/google.com/cloudsdktool/google-cloud-cli:588.0.0-alpine@sha256:44924be6dcf684d4e76922c5cb015950305e8ded844587f915dbaeefe290f545                             gcloud secrets versions access latest --secret="$sn" > "$stmp"
                    fi
                    if [ ! -s "$stmp" ]; then
                        echo "Secret $sn could not be retrieved or is empty"
                        exit 1
                    fi
                    mv "$stmp" "$sf"
                    chmod 400 "$sf"
                done
            fi
            sudo -u app docker-credential-gcr configure-docker --registries europe-west1-docker.pkg.dev
            sudo -u app docker run \
                --name workload_container \
                    -e A=1 \
    -e B=two \
                    -v /home/app/config.yaml:/config.yaml \
    -v /home/app/secrets/a-secret:/secrets/a-secret \
    -v /mnt/disks/work:/mnt/disks/work \
                --log-driver="gcplogs" \
                an-image \
                pts --step a_step 2>&1 | tr '\r' '\n' | awk '!/^ *[0-9]+%\|/'
