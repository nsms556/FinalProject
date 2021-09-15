import React, {useEffect, useState} from "react";
import {Card, CardHeader, Container, Row} from "reactstrap";

import axios from "axios";

import Header from "components/Headers/Header.js";
import SongInfo from "components/SongInfo.js";

import MakePlaylist from "./Sections/MakePlaylist";


function PlaylistPage() {

    const [SongList, setSongList] = useState(null);

    useEffect(() => {

        axios.get('http://127.0.0.1:8000/playlist/recommend')
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
    }, [SongList]);

    return (
        <>
            <Header/>
            <Container className="mt--7" fluid>
                <Row>
                    <div className="col">
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
                    </div>
                </Row>
            </Container>
        </>
    );
}

export default PlaylistPage;
