import React, {useEffect, useState} from "react";
import {Card, Container, Row} from "reactstrap";
import {trackPromise} from "react-promise-tracker";

import axios from "axios";

import Header from "components/Headers/Header.js";
import SongInfo from "components/SongInfo.js";

import SearchBox from "./Sections/SearchBox";


function SearchPage() {

    const [SongList, setSongList] = useState(null);

    const [LikeSongList, setLikeSongList] = useState(null);
    const [DislikeSongList, setDislikeSongList] = useState(null);

    useEffect(() => {

        trackPromise(
            axios.get('http://127.0.0.1:8000/users/')
                .then(response => {
                    if (response.data) {
                        setLikeSongList(response.data.like);
                        setDislikeSongList(response.data.dislike);
                    } else {
                        alert('Playlist(song_list)를 가져오지 못했습니다.');
                    }
                })
                .catch(error => {
                    console.log(error);
                })
        ).then(_ => {
            setLikeSongList(makeSongIdSet(LikeSongList));
            setDislikeSongList(makeSongIdSet(DislikeSongList));
        })
    }, []);

    const makeSongIdSet = (list) => {

        return list && list.map((song, _) => {
            return song.song_id;
        });
    };

    return (
        <>
            <Header/>
            <Container className="mt--7" fluid>
                <Row>
                    <div className="col">
                        <Card className="shadow">
                            <SearchBox getSongList={setSongList}/>
                            {
                                SongList ? (
                                    <SongInfo is_search={true} like_list={LikeSongList} dislike_list={DislikeSongList}
                                              song_list={SongList}/>) : null
                            }
                        </Card>
                    </div>
                </Row>
            </Container>
        </>
    );
}

export default SearchPage;
