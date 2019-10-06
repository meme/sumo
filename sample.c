#include <stdint.h>

// https://blog.quarkslab.com/deobfuscation-recovering-an-ollvm-protected-program.html
uint32_t target(uint32_t n) {
  uint32_t mod = n % 4;
  uint32_t result = 0;

  if (mod == 0) {
    result = (n | 0xbaaad0bf) * (2 ^ n);
  } else if (mod == 1) {
    result = (n & 0xbaaad0bf) * (3 + n);
  } else if (mod == 2) {
    result = (n ^ 0xbaaad0bf) * (4 | n);
  } else {
    result = (n + 0xbaaad0bf) * (5 & n);
  }

  return result;
}

int main() {
    // Test against the lifted execution
    printf("%d\n", target(10));
}