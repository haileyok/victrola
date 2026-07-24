#!/bin/sh
# Entrypoint: fix data directory ownership, then drop to the victrola user.
#
# Bind-mounted volumes (e.g. /mnt/user/appdata/victrola:/app/data) are owned
# by root on the host. The app runs as the non-root "victrola" user, so it
# can't write to SQLite or create files without this fixup. We chown the data
# dir as root, then use gosu to exec the real command as victrola.

set -e

DATA_DIR="${DATA_DIR:-data}"

# Resolve to absolute path (DATA_DIR is relative to /app by default)
case "$DATA_DIR" in
    /*) data_path="$DATA_DIR" ;;
    *)  data_path="/app/$DATA_DIR" ;;
esac

# Ensure the data directory exists and is owned by the victrola user.
mkdir -p "$data_path"
chown -R victrola:victrola "$data_path"

# Drop privileges and run the command.
exec gosu victrola "$@"
