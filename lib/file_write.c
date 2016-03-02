
#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>

#include "file_write.h"
#include "buffers.h"

void* file_write_thread(void * arg)
{
    struct fileWriteThreadArg * args = (struct fileWriteThreadArg *) arg;

    int fd;
    int useableBufferIDs[1] = {args->disk_ID};
    int bufferID = args->disk_ID;
    int file_num = args->disk_ID;
    int ret = 0;

    for (;;) {

        // This call is blocking.
        bufferID = get_full_buffer_from_list(args->buf, useableBufferIDs, 1);

        //printf("Got buffer, id: %d\n", bufferID);

        // Check if the producer has finished, and we should exit.
        if (bufferID == -1) {
            fprintf(stderr, "Exiting file write thread\n");
            pthread_exit((void *) &ret);
        }

        // Open the file to write

        const int file_name_len = 100;
        char file_name[file_name_len];

        if (args->num_disks == 1) {
            snprintf(file_name, file_name_len,
                     "%s/%s/%d_%07d.dat",
                     args->disk_base,
                     args->dataset_name,
                     args->link_ID,
                     file_num);
        } else {
            snprintf(file_name, file_name_len,
                     "%s/%d/%s/%d_%07d.dat",
                     args->disk_base,
                     args->disk_ID,
                     args->dataset_name,
                     args->link_ID,
                     file_num);
        }

        fd = open(file_name, O_WRONLY | O_CREAT, 0666);

        if (fd == -1) {
            perror("Cannot open file");
            fprintf(stderr, "File name was: %s", file_name);
            exit(errno);
        }

        ssize_t bytes_writen = write(fd, args->buf->data[bufferID], args->buf->buffer_size);

        if (bytes_writen != args->buf->buffer_size) {
            printf("Failed to write buffer to disk!!!  Abort, Panic, etc.");
            exit(-1);
        } else {
             fprintf(stderr, "Data writen to file: %s\n", file_name);
        }

        if (close(fd) == -1) {
            fprintf(stderr, "Cannot close file");
        }

        mark_buffer_empty(args->buf, bufferID);

        useableBufferIDs[0] = ( useableBufferIDs[0] + args->num_disks ) % ( args->buf->num_buffers );

        file_num += args->num_disks;
    }

}