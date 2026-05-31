Data sharing options (do NOT commit raw data to this repository)

Recommended approaches for sharing NIfTI/DICOM data with collaborators:

1) Create an archive locally and share
   - Create a compressed tarball of the data directory (keeps permissions and structure):
     tar -czf data_for_jacob.tar.gz -C /path/to/dataset .
   - Use a secure file transfer (scp) or a cloud storage link to share the tarball.

2) Use rsync over SSH for large datasets
   - rsync -avh --progress /path/to/dataset jacob@example.org:/path/to/destination
   - This resumes interrupted transfers and only sends diffs on repeated syncs.

3) Upload to a cloud bucket or file-share and share access
   - S3: use 'aws s3 cp' or 'aws s3 sync' to upload; share pre-signed URLs or IAM-limited access.
   - Google Drive/Dropbox: create a shared folder and drop the archive there.

4) Create a minimal, anonymized subset for inclusion in the repo
   - Use dcm2niix or nibabel to convert and strip PHI, then provide a tiny sample (a single slice) in 'examples/data_sample' if absolutely needed.

Verification steps after transfer:
   - Confirm checksums (sha256sum) match between sender and receiver.
   - Ensure recipient understands data usage, licensing, and PHI constraints.

If you want, I can create example commands or a small helper script to prepare the archive and compute checksums.