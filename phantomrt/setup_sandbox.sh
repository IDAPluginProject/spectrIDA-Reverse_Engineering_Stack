#!/bin/bash
# Atlas Sandbox Setup — Run this in WSL
# Sets up the exploration environment

echo "Setting up Atlas Sandbox..."

# Install essentials
apt-get update -qq
apt-get install -y -qq gcc python3 python3-pip strace ltrace file 2>/dev/null

# Create workspace
mkdir -p /tmp/atlas/{programs,traces,logs}

# ═══════════════════════════════════════
# VULNERABLE C PROGRAMS (for crash learning)
# ═══════════════════════════════════════

cat > /tmp/atlas/programs/overflow.c << 'EOF'
#include <stdio.h>
#include <string.h>
void vulnerable() {
    char buffer[32];
    printf("Enter data: ");
    gets(buffer);  // DANGEROUS: no bounds check
    printf("You said: %s\n", buffer);
}
int main() { vulnerable(); return 0; }
EOF

cat > /tmp/atlas/programs/format_str.c << 'EOF'
#include <stdio.h>
void vulnerable(char *input) {
    printf(input);  // DANGEROUS: user controls format
}
int main(int argc, char **argv) {
    if(argc > 1) vulnerable(argv[1]);
    else printf("Usage: %s <input>\n", argv[0]);
    return 0;
}
EOF

cat > /tmp/atlas/programs/buffer_heap.c << 'EOF'
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
void vulnerable() {
    char *heap_buf = malloc(16);
    char input[64];
    printf("Enter: ");
    fgets(input, 64, stdin);
    strcpy(heap_buf, input);  // DANGEROUS: heap overflow
    printf("Stored: %s\n", heap_buf);
    free(heap_buf);
}
int main() { vulnerable(); return 0; }
EOF

cat > /tmp/atlas/programs/safe_calc.c << 'EOF'
#include <stdio.h>
#include <stdlib.h>
int main(int argc, char **argv) {
    if(argc != 3) { printf("Usage: %s <a> <b>\n", argv[0]); return 1; }
    int a = atoi(argv[1]);
    int b = atoi(argv[2]);
    printf("%d + %d = %d\n", a, b, a + b);
    printf("%d * %d = %d\n", a, b, a * b);
    return 0;
}
EOF

cat > /tmp/atlas/programs/string_tool.c << 'EOF'
#include <stdio.h>
#include <string.h>
void reverse(char *s) {
    int len = strlen(s);
    for(int i = 0; i < len/2; i++) {
        char tmp = s[i];
        s[i] = s[len-1-i];
        s[len-1-i] = tmp;
    }
}
int main(int argc, char **argv) {
    if(argc > 1) {
        reverse(argv[1]);
        printf("Reversed: %s\n", argv[1]);
    }
    return 0;
}
EOF

cat > /tmp/atlas/programs/file_reader.c << 'EOF'
#include <stdio.h>
#include <stdlib.h>
int main(int argc, char **argv) {
    if(argc != 2) { printf("Usage: %s <filename>\n", argv[0]); return 1; }
    FILE *f = fopen(argv[1], "r");
    if(!f) { printf("Cannot open %s\n", argv[1]); return 1; }
    char buf[256];
    while(fgets(buf, sizeof(buf), f)) printf("%s", buf);
    fclose(f);
    return 0;
}
EOF

# Compile all programs
echo "Compiling programs..."
for f in /tmp/atlas/programs/*.c; do
    name=$(basename "$f" .c)
    gcc -o "/tmp/atlas/programs/$name" "$f" -z execstack 2>/dev/null
    echo "  Compiled: $name"
done

# Create test files for file_reader
echo "Hello from Atlas!" > /tmp/atlas/test_file.txt
echo "This is a test file with some data." >> /tmp/atlas/test_file.txt
echo "Line 3 of the file." >> /tmp/atlas/test_file.txt

echo ""
echo "Sandbox ready!"
echo "Programs: $(ls /tmp/atlas/programs/*.c | wc -l) vulnerable C programs"
echo "Location: /tmp/atlas/"
echo ""
echo "Programs available:"
ls -la /tmp/atlas/programs/
