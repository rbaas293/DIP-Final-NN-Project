## Generating a new SSH key

* `ssh-keygen -t rsa -b 4096 -C "your_email@example.com"`
* or use the following for increased rsa length and security:
  - `ssh-keygen -t rsa -b 15360 -C "your_email@example.com"`
* When you're prompted to "Enter a file in which to save the key," press Enter. This accepts the default file location. and will overwrite it if the file already exisits.
* At the prompt, type a secure passphrase or just leave it empty. Just be aware that if anyone gets ahold of your private key, your screwed. 
> Note: If you are interest in how ssh-kegen works to generate a secure rsa key, check out the following link.
> https://crypto.stackexchange.com/questions/66076/how-to-efficiently-generate-a-random-safe-prime-of-given-length
