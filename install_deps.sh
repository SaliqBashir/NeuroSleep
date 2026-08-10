#!/bin/bash
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
cd /home/saliq/NeuroSleep/frontend
npm install
npm install recharts lucide-react axios
