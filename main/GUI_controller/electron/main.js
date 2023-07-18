const { app, BrowserWindow, ipcMain } = require('electron')
const net = require('net');
let client;

function createWindow () {

    const win = new BrowserWindow({
        fullscreen: true,
        webPreferences: {
            nodeIntegration: true,
            contextIsolation: false
        }
    })

    win.loadFile('index.html')

    win.webContents.on('did-finish-load', () => {
        client = net.createConnection('/home/pi/uds_socket', () => {
            // console.log('Connected to server');
            // client.write('Hello from Node.js');
        });

        client.on('data', (data) => {
            // console.log(data.toString());
            win.webContents.send('uds_data', data.toString());
        });
    
        client.on('end', () => {
            // console.log('Disconnected from server');
        });
    });

    ipcMain.on('message_to_python', (event, data) => {
        // console.log('Received message from renderer: ', data);
        if (client) { // Check if client is not null
            client.write(data);
        }
    });
}

app.whenReady().then(createWindow)
