import React, {useEffect, useState} from "react";
import {Card, CardHeader, Container, Row} from "reactstrap";
import {trackPromise} from 'react-promise-tracker';

import axios from "axios";

import Header from "components/Headers/Header.js";
import LoadingIndicator from "components/LoadingIndicator";
import SongInfo from "components/SongInfo.js";

import MakePlaylist from "./Sections/MakePlaylist";

axios.defaults.withCredentials = true;

function PlaylistPage() {

    const [SongList, setSongList] = useState(null);
    const [ViewPage, setViewPage] = useState(false);

    useEffect(() => {
        trackPromise(
            axios.get('http://127.0.0.1:8000/playlist/songs')
                .then(response => {
                    if (response.data) {
                        setSongList(response.data.song_list);
                    } else {
                        alert('Playlist(song_list)를 가져오지 못했습니다.');
                    }
                })
                .catch(error => {
                    console.log(error);
                })
        ).then(_ => {
            setViewPage(true);
        })
    }, [SongList]);

    return (
        <>
            <Header/>
            <Container className="mt--7" fluid>
                <Row>
                    <div className="col">
                        <LoadingIndicator/>
                        {
                            ViewPage &&
                            <Card className="shadow">
                                {
                                    SongList ?
                                        (
                                            <>
                                                <CardHeader className="border-0">
                                                    <h3 className="mb-0">My Playlist</h3>
                                                </CardHeader>
                                                <SongInfo song_list={SongList}/>
                                            </>
                                        ) :
                                        (
                                            <MakePlaylist/>
                                        )
                                }
                            </Card>
                        }
                    </div>
                </Row>
            </Container>
        </>
    );
}

export default PlaylistPage;
